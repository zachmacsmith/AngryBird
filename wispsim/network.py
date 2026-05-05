"""
network.py

Ping-based drone mesh network layer for WISP/Wisp simulations.

Place this file next to runner.py, for example:
    simulation/network.py

Purpose:
    Drones still measure observations normally, but measured data is not
    immediately available to the ground station. Each drone stores measured
    observations in an onboard buffer. A lightweight ping/heartbeat model
    estimates live link quality between nodes. The network builds a graph
    from fresh links, uses Dijkstra's algorithm to find the best path to the
    ground station, then transmits buffered packets only when a valid path
    exists.

The best path is defined as the path with the lowest communication cost.
By default, communication cost is based mainly on link success probability:
    cost = -log(success_probability) + hop_penalty + latency_weight * latency

This makes a path with several strong short links better than one weak long link.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
import uuid
from typing import Optional

import numpy as np

from angrybird.types import DroneObservation


GROUND_STATION_ID = "GS"


@dataclass
class TelemetryPacket:
    """A group of drone observations waiting to be delivered."""

    packet_id: str
    drone_id: str
    created_time: float
    observations: list[DroneObservation]
    priority: int = 2
    retry_count: int = 0
    delivered: bool = False


@dataclass
class DroneBuffer:
    """Onboard buffer for one drone."""

    drone_id: str
    packets: list[TelemetryPacket] = field(default_factory=list)

    def add_packet(self, packet: TelemetryPacket) -> None:
        if len(packet.observations) > 0:
            self.packets.append(packet)

    def get_send_candidates(self, max_packets: int) -> list[TelemetryPacket]:
        # Lower priority number means more urgent.
        self.packets.sort(key=lambda packet: (packet.priority, -packet.created_time))
        return self.packets[:max_packets]

    def remove_packets(self, packet_ids: set[str]) -> None:
        remaining: list[TelemetryPacket] = []

        for i in range(len(self.packets)):
            packet = self.packets[i]

            if packet.packet_id not in packet_ids:
                remaining.append(packet)

        self.packets = remaining

    def size(self) -> int:
        return len(self.packets)


@dataclass
class LinkState:
    """Estimated live link state from heartbeat pings."""

    neighbor_id: str
    quality: float
    success_probability: float
    latency_s: float
    last_seen_time: float


class NeighborTable:
    """Live neighbor table for one node."""

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.links: dict[str, LinkState] = {}

    def update_link(
        self,
        neighbor_id: str,
        quality: float,
        success_probability: float,
        latency_s: float,
        current_time: float,
    ) -> None:
        self.links[neighbor_id] = LinkState(
            neighbor_id=neighbor_id,
            quality=float(quality),
            success_probability=float(success_probability),
            latency_s=float(latency_s),
            last_seen_time=float(current_time),
        )

    def get_fresh_links(
        self,
        current_time: float,
        max_age_s: float,
    ) -> dict[str, LinkState]:
        fresh: dict[str, LinkState] = {}

        for neighbor_id in self.links:
            link = self.links[neighbor_id]

            if current_time - link.last_seen_time <= max_age_s:
                fresh[neighbor_id] = link

        return fresh


@dataclass
class MeshNetworkConfig:
    mesh_range_m: float = 1800.0
    max_link_age_s: float = 45.0
    min_link_quality: float = 0.12
    ping_interval_s: float = 10.0

    # Keep a small hop penalty so the route does not take unnecessary hops.
    # But reliability still matters more than hop count.
    hop_penalty: float = 0.05
    latency_weight: float = 0.0

    # Realistic improved telemetry:
    # enough throughput for store-and-forward, not unlimited bandwidth.
    max_packets_per_drone_per_tick: int = 8
    background_packet_loss_probability: float = 0.015

    # Smooth ping quality so routes do not jump too wildly every timestep.
    quality_smoothing_alpha: float = 0.70

    # Stale non-urgent packets should not dominate real-time wildfire prediction.
    # 300s = 5 minutes.
    max_packet_age_s: float = 300.0

    relay_id: Optional[str] = None
    relay_range_m: Optional[float] = None

    retransmit_attempts_per_packet: int = 2
    urgent_retransmit_attempts_per_packet: int = 3

def make_pams_like_mesh_config() -> MeshNetworkConfig:
        return MeshNetworkConfig(
            mesh_range_m=1200.0,
            max_link_age_s=30.0,
            min_link_quality=0.20,
            ping_interval_s=10.0,
            hop_penalty=0.05,
            latency_weight=0.0,
            max_packets_per_drone_per_tick=3,
            background_packet_loss_probability=0.03,
            quality_smoothing_alpha=0.70,
            max_packet_age_s=720.0,
            relay_id=None,
            relay_range_m=None,
        )

def make_improved_mesh_config() -> MeshNetworkConfig:
        return MeshNetworkConfig(
            mesh_range_m=1800.0,
            max_link_age_s=45.0,
            min_link_quality=0.12,
            ping_interval_s=10.0,
            hop_penalty=0.05,
            latency_weight=0.0,
            max_packets_per_drone_per_tick=8,
            background_packet_loss_probability=0.015,
            quality_smoothing_alpha=0.70,
            max_packet_age_s=1000.0,
            relay_id=None,
            relay_range_m=None,
        )

@dataclass
class MeshNetworkMetrics:
    packets_created: int = 0
    packets_delivered: int = 0
    packets_failed: int = 0
    observations_created: int = 0
    observations_delivered: int = 0
    total_delay_s: float = 0.0
    delivered_delays_s: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, float]:
        packet_delivery_rate = 0.0
        observation_delivery_rate = 0.0
        mean_delay_s = 0.0

        if self.packets_created > 0:
            packet_delivery_rate = self.packets_delivered / self.packets_created

        if self.observations_created > 0:
            observation_delivery_rate = (
                self.observations_delivered / self.observations_created
            )

        if len(self.delivered_delays_s) > 0:
            mean_delay_s = self.total_delay_s / len(self.delivered_delays_s)

        return {
            "packets_created": float(self.packets_created),
            "packets_delivered": float(self.packets_delivered),
            "packets_failed": float(self.packets_failed),
            "packet_delivery_rate": float(packet_delivery_rate),
            "observations_created": float(self.observations_created),
            "observations_delivered": float(self.observations_delivered),
            "observation_delivery_rate": float(observation_delivery_rate),
            "mean_delivery_delay_s": float(mean_delay_s),
        }


class PingMeshNetwork:
    """
    Ping-based self-healing mesh network.

    Main methods used by runner.py:
        buffer_observations(...)
        step(...)

    Typical runner workflow:
        1. Drone collects observations.
        2. Call network.buffer_observations(drone_id, current_time, observations).
        3. After all drones moved and measured, call:
               received_obs = network.step(drone_positions, current_time)
        4. Add only received_obs to obs_buffer and LiveEstimator.
    """

    def __init__(
        self,
        ground_station_position: np.ndarray,
        drone_ids: list[str],
        config: Optional[MeshNetworkConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.ground_station_position = np.asarray(
            ground_station_position,
            dtype=np.float64,
        )

        self.config = config if config is not None else MeshNetworkConfig()
        self.rng = rng if rng is not None else np.random.default_rng()

        self.buffers: dict[str, DroneBuffer] = {}
        self.neighbor_tables: dict[str, NeighborTable] = {}

        self.neighbor_tables[GROUND_STATION_ID] = NeighborTable(GROUND_STATION_ID)

        for i in range(len(drone_ids)):
            drone_id = drone_ids[i]
            self.register_drone(drone_id)

        if self.config.relay_id is not None:
            self._ensure_node(self.config.relay_id)

        self.metrics = MeshNetworkMetrics()
        self.last_paths: dict[str, Optional[list[str]]] = {}
        self.last_connected_drones: set[str] = set()
        self.last_graph: dict[str, list[tuple[str, float]]] = {}

        self._last_ping_time: Optional[float] = None

    def register_drone(self, drone_id: str) -> None:
        if drone_id not in self.buffers:
            self.buffers[drone_id] = DroneBuffer(drone_id)

        self._ensure_node(drone_id)

    def _ensure_node(self, node_id: str) -> None:
        if node_id not in self.neighbor_tables:
            self.neighbor_tables[node_id] = NeighborTable(node_id)

    def make_packet(
        self,
        drone_id: str,
        current_time: float,
        observations: list[DroneObservation],
        priority: int = 2,
    ) -> TelemetryPacket:
        return TelemetryPacket(
            packet_id=str(uuid.uuid4()),
            drone_id=drone_id,
            created_time=float(current_time),
            observations=observations,
            priority=int(priority),
        )

    def buffer_observations(
        self,
        drone_id: str,
        current_time: float,
        observations: list[DroneObservation],
        priority: int = 2,
    ) -> None:
        if len(observations) == 0:
            return

        self.register_drone(drone_id)

        packet = self.make_packet(
            drone_id=drone_id,
            current_time=current_time,
            observations=observations,
            priority=priority,
        )

        self.buffers[drone_id].add_packet(packet)
        self.metrics.packets_created += 1
        self.metrics.observations_created += len(observations)

    def step(
        self,
        drone_positions: dict[str, np.ndarray],
        current_time: float,
        relay_position: Optional[np.ndarray] = None,
    ) -> list[DroneObservation]:
        """
        Update pings, route packets, and return observations received by GS.

        drone_positions should map drone_id to position arrays in metres.
        Position convention matches drone_sim.py: [y_m, x_m].
        """

        node_positions = self._build_node_positions(drone_positions, relay_position)

        should_ping = False

        if self._last_ping_time is None:
            should_ping = True
        elif current_time - self._last_ping_time >= self.config.ping_interval_s:
            should_ping = True

        if should_ping:
            self.update_neighbor_tables_with_pings(
                node_positions=node_positions,
                current_time=current_time,
            )
            self._last_ping_time = float(current_time)

        graph = self.build_graph_from_neighbor_tables(current_time=current_time)
        self.last_graph = graph

        received_observations: list[DroneObservation] = []
        self.last_paths = {}
        self.last_connected_drones = set()

        drone_ids = list(self.buffers.keys())

        for i in range(len(drone_ids)):
            drone_id = drone_ids[i]
            path = self.find_best_path_to_ground(graph, drone_id)
            self.last_paths[drone_id] = path

            if path is None:
                continue

            self.last_connected_drones.add(drone_id)

            delivered_packet_ids: set[str] = set()
            self._drop_stale_packets(drone_id, current_time)

            def _drop_stale_packets(self, drone_id: str, current_time: float) -> None:
                if drone_id not in self.buffers:
                    return

                fresh_packets = []

                for i in range(len(self.buffers[drone_id].packets)):
                    packet = self.buffers[drone_id].packets[i]
                    age_s = current_time - packet.created_time

                    # Always keep urgent packets.
                    if packet.priority == 1:
                        fresh_packets.append(packet)

                    # Keep normal packets only if they are still useful for real-time prediction.
                    elif age_s <= self.config.max_packet_age_s:
                        fresh_packets.append(packet)

                self.buffers[drone_id].packets = fresh_packets

            candidates = self.buffers[drone_id].get_send_candidates(
                self.config.max_packets_per_drone_per_tick
            )

            for j in range(len(candidates)):
                packet = candidates[j]

                if packet.priority == 1:
                    max_attempts = self.config.urgent_retransmit_attempts_per_packet
                else:
                    max_attempts = self.config.retransmit_attempts_per_packet

                delivered = False

                for attempt in range(max_attempts):
                    packet.retry_count += 1

                    if self._try_transmit_packet_along_path(packet, path):
                        delivered = True
                        break

                if delivered:
                    packet.delivered = True
                    delivered_packet_ids.add(packet.packet_id)
                    received_observations.extend(packet.observations)

                    delay_s = float(current_time - packet.created_time)
                    self.metrics.packets_delivered += 1
                    self.metrics.observations_delivered += len(packet.observations)
                    self.metrics.total_delay_s += delay_s
                    self.metrics.delivered_delays_s.append(delay_s)
                else:
                    self.metrics.packets_failed += 1

            self.buffers[drone_id].remove_packets(delivered_packet_ids)

        return received_observations

    def _build_node_positions(
        self,
        drone_positions: dict[str, np.ndarray],
        relay_position: Optional[np.ndarray],
    ) -> dict[str, np.ndarray]:
        node_positions: dict[str, np.ndarray] = {}

        for drone_id in drone_positions:
            node_positions[drone_id] = np.asarray(drone_positions[drone_id], dtype=np.float64)
            self.register_drone(drone_id)

        node_positions[GROUND_STATION_ID] = self.ground_station_position

        if self.config.relay_id is not None and relay_position is not None:
            node_positions[self.config.relay_id] = np.asarray(
                relay_position,
                dtype=np.float64,
            )
            self._ensure_node(self.config.relay_id)

        return node_positions

    def update_neighbor_tables_with_pings(
        self,
        node_positions: dict[str, np.ndarray],
        current_time: float,
    ) -> None:
        node_ids = list(node_positions.keys())

        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                a_id = node_ids[i]
                b_id = node_ids[j]

                self._ensure_node(a_id)
                self._ensure_node(b_id)

                a_pos = node_positions[a_id]
                b_pos = node_positions[b_id]

                max_range_m = self._range_for_pair(a_id, b_id)
                ping = self._simulate_ping(a_pos, b_pos, max_range_m)

                if ping is None:
                    continue

                quality, success_probability, latency_s = ping

                quality_ab = self._smooth_existing_quality(a_id, b_id, quality)
                quality_ba = self._smooth_existing_quality(b_id, a_id, quality)

                self.neighbor_tables[a_id].update_link(
                    neighbor_id=b_id,
                    quality=quality_ab,
                    success_probability=success_probability,
                    latency_s=latency_s,
                    current_time=current_time,
                )

                self.neighbor_tables[b_id].update_link(
                    neighbor_id=a_id,
                    quality=quality_ba,
                    success_probability=success_probability,
                    latency_s=latency_s,
                    current_time=current_time,
                )

    def _range_for_pair(self, a_id: str, b_id: str) -> float:
        relay_id = self.config.relay_id

        if relay_id is not None:
            if a_id == relay_id or b_id == relay_id:
                if self.config.relay_range_m is not None:
                    return float(self.config.relay_range_m)

        return float(self.config.mesh_range_m)

    def _simulate_ping(
        self,
        pos_a: np.ndarray,
        pos_b: np.ndarray,
        max_range_m: float,
    ) -> Optional[tuple[float, float, float]]:
        success_probability = self._estimate_link_success_probability(
            pos_a=pos_a,
            pos_b=pos_b,
            max_range_m=max_range_m,
        )

        if success_probability <= 0.0:
            return None

        ping_succeeded = self.rng.random() <= success_probability

        if not ping_succeeded:
            return None

        d = self._distance_m(pos_a, pos_b)
        latency_s = 0.01 + 0.00002 * d
        quality = success_probability

        return quality, success_probability, latency_s

    def _estimate_link_success_probability(
        self,
        pos_a: np.ndarray,
        pos_b: np.ndarray,
        max_range_m: float,
    ) -> float:
        d = self._distance_m(pos_a, pos_b)

        if d > max_range_m:
            return 0.0

        ratio = d / max(max_range_m, 1e-9)

        # Simple wireless degradation model:
        # close nodes have high success probability, edge-of-range nodes are weak.
        probability = 1.0 - ratio ** 2

        if probability < 0.05:
            probability = 0.05

        if probability > 0.99:
            probability = 0.99

        return float(probability)

    def _smooth_existing_quality(
        self,
        node_id: str,
        neighbor_id: str,
        new_quality: float,
    ) -> float:
        table = self.neighbor_tables[node_id]
        alpha = self.config.quality_smoothing_alpha

        if neighbor_id not in table.links:
            return float(new_quality)

        old_quality = table.links[neighbor_id].quality
        return float(alpha * old_quality + (1.0 - alpha) * new_quality)

    def build_graph_from_neighbor_tables(
        self,
        current_time: float,
    ) -> dict[str, list[tuple[str, float]]]:
        graph: dict[str, list[tuple[str, float]]] = {}

        for node_id in self.neighbor_tables:
            graph[node_id] = []

        for node_id in self.neighbor_tables:
            fresh_links = self.neighbor_tables[node_id].get_fresh_links(
                current_time=current_time,
                max_age_s=self.config.max_link_age_s,
            )

            for neighbor_id in fresh_links:
                link = fresh_links[neighbor_id]

                if link.quality < self.config.min_link_quality:
                    continue

                success_probability = max(1e-6, link.success_probability)

                cost = (
                    -math.log(success_probability)
                    + self.config.hop_penalty
                    + self.config.latency_weight * link.latency_s
                )

                graph[node_id].append((neighbor_id, float(cost)))

        return graph

    def find_best_path_to_ground(
        self,
        graph: dict[str, list[tuple[str, float]]],
        start_id: str,
    ) -> Optional[list[str]]:
        if start_id not in graph:
            return None

        distances: dict[str, float] = {}
        previous: dict[str, Optional[str]] = {}

        for node_id in graph:
            distances[node_id] = float("inf")
            previous[node_id] = None

        distances[start_id] = 0.0

        pq: list[tuple[float, str]] = []
        heapq.heappush(pq, (0.0, start_id))

        while len(pq) > 0:
            current_cost, current_id = heapq.heappop(pq)

            if current_id == GROUND_STATION_ID:
                break

            if current_cost > distances[current_id]:
                continue

            neighbors = graph[current_id]

            for i in range(len(neighbors)):
                neighbor_id, edge_cost = neighbors[i]
                new_cost = current_cost + edge_cost

                if new_cost < distances.get(neighbor_id, float("inf")):
                    distances[neighbor_id] = new_cost
                    previous[neighbor_id] = current_id
                    heapq.heappush(pq, (new_cost, neighbor_id))

        if distances.get(GROUND_STATION_ID, float("inf")) == float("inf"):
            return None

        path: list[str] = []
        current: Optional[str] = GROUND_STATION_ID

        while current is not None:
            path.append(current)
            current = previous[current]

        path.reverse()
        return path

    def _try_transmit_packet_along_path(
        self,
        packet: TelemetryPacket,
        path: list[str],
    ) -> bool:
        if len(path) < 2:
            return False

        # Background loss models UDP-like drops not explained by distance alone.
        if self.rng.random() < self.config.background_packet_loss_probability:
            return False

        for i in range(len(path) - 1):
            a_id = path[i]
            b_id = path[i + 1]

            success_probability = self._get_link_success_probability(a_id, b_id)

            if success_probability <= 0.0:
                return False

            if self.rng.random() > success_probability:
                return False

        return True

    def _get_link_success_probability(self, a_id: str, b_id: str) -> float:
        if a_id not in self.neighbor_tables:
            return 0.0

        table = self.neighbor_tables[a_id]

        if b_id not in table.links:
            return 0.0

        return float(table.links[b_id].success_probability)

    def _distance_m(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def get_metrics(self) -> dict[str, float]:
        return self.metrics.as_dict()

    def get_buffer_sizes(self) -> dict[str, int]:
        sizes: dict[str, int] = {}

        for drone_id in self.buffers:
            sizes[drone_id] = self.buffers[drone_id].size()

        return sizes

    def get_last_paths(self) -> dict[str, Optional[list[str]]]:
        return self.last_paths

    def get_connected_drones(self) -> set[str]:
        return set(self.last_connected_drones)

    def _drop_stale_packets(self, drone_id: str, current_time: float) -> None:
        if drone_id not in self.buffers:
            return

        fresh_packets = []

        for i in range(len(self.buffers[drone_id].packets)):
            packet = self.buffers[drone_id].packets[i]
            age_s = current_time - packet.created_time

            if packet.priority == 1:
                fresh_packets.append(packet)
            elif age_s <= self.config.max_packet_age_s:
                fresh_packets.append(packet)

        self.buffers[drone_id].packets = fresh_packets


def assign_packet_priority(observations: list[DroneObservation]) -> int:
    """
    Simple starter priority function.

    Lower number means higher priority:
        1 = urgent
        2 = normal
        3 = low priority

    You can later replace this with fire-front detection, battery emergency,
    or sudden wind/FMC change logic.
    """

    for i in range(len(observations)):
        obs = observations[i]

        if obs.wind_speed is not None:
            if np.isfinite(obs.wind_speed) and obs.wind_speed >= 12.0:
                return 1

    return 2
