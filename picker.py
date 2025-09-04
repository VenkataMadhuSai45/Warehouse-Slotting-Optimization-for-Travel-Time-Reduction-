import pandas as pd
import numpy as np
import math
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vrp_routing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- 1. CONFIGURATION ---

# File Paths
SLOTTING_RESULTS_FILE = 'slotting_assignments_output.csv'
ORDER_FILE_PATH = 'order_batch.csv'
# NEW: Filename for the summary CSV output
ROUTE_SUMMARY_FILE = 'picker_summary_by_route.csv'

# Warehouse Layout Configuration (Must match slotting script)
WAREHOUSE_PARAMS = {
    'racks': 30, 'bays': 24, 'levels': 7, 'middle_aisle_after_bay': 11,
    'bay_width': 1.0, 'rack_depth': 1.0, 'picking_aisle_width': 3.0, 'cross_aisle_width': 4.0,
}
DISPATCH_ZONE_LOCATION = 'front'

# Picker and Order Parameters
PICKER_CONFIG = {
    'num_pickers': 2,
    'picker_capacity': 50,  # Increased capacity
    'picker_speed': 3.5,
    'base_picking_time': 0.4,
    'time_penalty_per_level': 0.1,
    'max_tour_time': 60,  # Increased time
    'setup_time': 1.0,
}

# OR-Tools Configuration
SOLVER_CONFIG = {
    'time_limit_seconds': 30,
    'solution_limit': 100,
    'local_search_metaheuristic': 'GUIDED_LOCAL_SEARCH',
    'first_solution_strategy': 'PATH_CHEAPEST_ARC',
}


def load_and_consolidate_orders(file_path: str) -> Optional[Tuple[List[str], List[int]]]:
    """Loads and consolidates orders from a CSV file into a single pick wave."""
    logger.info(f"Loading and consolidating orders from '{file_path}'...")
    try:
        order_df = pd.read_csv(file_path)
        required_cols = ['OrderID', 'SKU', 'Quantity']
        if not all(col in order_df.columns for col in required_cols):
            raise ValueError(f"Order file must contain columns: {required_cols}")

        consolidated = order_df.groupby('SKU')['Quantity'].sum().reset_index()
        consolidated = consolidated[consolidated['Quantity'] > 0]
        logger.info(f"[SUCCESS] Consolidated into a single pick wave with {len(consolidated)} unique SKUs.")
        return consolidated['SKU'].astype(str).tolist(), consolidated['Quantity'].astype(int).tolist()

    except FileNotFoundError:
        logger.error(f"[ERROR] Order file not found: {file_path}")
        logger.info(f"Creating a sample 'order_batch.csv' file. Please edit it with your order data.")
        sample_data = pd.DataFrame({
            'OrderID': ['ORD101', 'ORD101', 'ORD102', 'ORD102', 'ORD103'],
            'SKU': ["M16280220", "M16280295", "M2N3G1450", "M16280220", "M16280427"],
            'Quantity': [5, 2, 8, 3, 10]
        })
        sample_data.to_csv(file_path, index=False)
        return None
    except Exception as e:
        logger.error(f"[ERROR] Failed to process order file: {e}")
        return None


def validate_warehouse_config(params: Dict[str, Any]) -> bool:
    try:
        for param in ['racks', 'bays', 'levels', 'middle_aisle_after_bay', 'bay_width', 'rack_depth',
                      'picking_aisle_width', 'cross_aisle_width']:
            if params.get(param) is None or params[param] <= 0: raise ValueError(
                f"Parameter '{param}' must be a positive number")
        if params['middle_aisle_after_bay'] >= params['bays']: raise ValueError(
            "Middle aisle position must be less than total bays")
        logger.info("[SUCCESS] Warehouse configuration validated successfully");
        return True
    except (ValueError, KeyError) as e:
        logger.error(f"[ERROR] Warehouse configuration validation failed: {e}");
        return False


def validate_picker_config(config: Dict[str, Any]) -> bool:
    try:
        for param in ['num_pickers', 'picker_capacity', 'picker_speed', 'max_tour_time', 'base_picking_time',
                      'time_penalty_per_level']:
            if config.get(param) is None or config[param] <= 0: raise ValueError(
                f"Picker config parameter '{param}' must be a positive number")
        logger.info("[SUCCESS] Picker configuration validated successfully");
        return True
    except (ValueError, KeyError) as e:
        logger.error(f"[ERROR] Picker configuration validation failed: {e}");
        return False


class WarehouseCoordinateSystem:
    def __init__(self, params: Dict[str, Any], dispatch_location: str):
        self.params = params
        self.total_width = (params['racks'] / 2) * (2 * params['rack_depth'] + params['picking_aisle_width'])
        self.total_length = params['bays'] * params['bay_width'] + params['cross_aisle_width']
        self.slot_coords = self._generate_slot_coordinates()
        self.dispatch_coords = self._get_dispatch_zone_coords(dispatch_location)

    def _generate_slot_coordinates(self) -> Dict[str, Tuple[float, float]]:
        slot_coords = {}
        for r in range(1, self.params['racks'] + 1):
            for b in range(1, self.params['bays'] + 1):
                rack_pair_index = (r - 1) // 2
                x = rack_pair_index * (2 * self.params['rack_depth'] + self.params['picking_aisle_width'])
                if (r - 1) % 2 == 1: x += self.params['rack_depth'] + self.params['picking_aisle_width']
                y = (b - 1) * self.params['bay_width']
                if b > self.params['middle_aisle_after_bay']: y += self.params['cross_aisle_width']
                for l in range(1, self.params['levels'] + 1):
                    slot_coords[f"R{r:02d}-B{b:02d}-L{l:02d}"] = (x, y)
        return slot_coords

    def _get_dispatch_zone_coords(self, location: str) -> Tuple[float, float]:
        if location == 'front':
            return (self.total_width / 2, self.total_length + self.params['cross_aisle_width'] / 2)
        else:
            logger.warning(f"Unknown dispatch zone '{location}', defaulting to 'front'.")
            return (self.total_width / 2, self.total_length + self.params['cross_aisle_width'] / 2)


class SlottingDataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_assignments(self) -> Optional[Dict[str, Dict[str, Any]]]:
        try:
            if not Path(self.file_path).exists(): raise FileNotFoundError(
                f"Slotting results file not found: {self.file_path}")
            logger.info(f"Loading slotting assignments from '{self.file_path}'...")
            slotting_df = pd.read_csv(self.file_path, dtype={'SKU': str})
            required_columns = ['SKU', 'Assigned_Slot', 'level']
            if not all(col in slotting_df.columns for col in required_columns):
                raise ValueError(f"Missing required columns in slotting file: {required_columns}")
            sku_to_details = {row['SKU']: {'slot': row['Assigned_Slot'], 'level': int(row['level'])} for _, row in
                              slotting_df.iterrows()}
            logger.info(f"[SUCCESS] Loaded assignment details for {len(sku_to_details)} SKUs")
            return sku_to_details
        except Exception as e:
            logger.error(f"[ERROR] Error loading slotting assignments: {e}");
            return None


def calculate_manhattan_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


class VRPSolver:
    def __init__(self, picker_config: Dict[str, Any], solver_config: Dict[str, Any]):
        self.picker_config = picker_config
        self.solver_config = solver_config
        self.time_precision_scaler = 100

    def solve_picker_vrp(self, order_skus: List[str], order_demands: List[int],
                         sku_to_details: Dict[str, Dict[str, Any]], slot_coords: Dict[str, Tuple[float, float]],
                         dispatch_coords: Tuple[float, float]) -> Optional[Dict[str, Any]]:
        logger.info(f"Solving VRP for batch of {len(order_skus)} unique SKUs...")

        node_locations, demands, service_times, node_details_list = [dispatch_coords], [0], [0], [{'type': 'depot'}]
        for sku, demand in zip(order_skus, order_demands):
            if sku in sku_to_details:
                details = sku_to_details[sku]
                if details['slot'] in slot_coords:
                    node_locations.append(slot_coords[details['slot']])
                    demands.append(demand)
                    service_time = self.picker_config['base_picking_time'] + (details['level'] - 1) * \
                                   self.picker_config['time_penalty_per_level']
                    service_times.append(service_time)
                    node_details_list.append({'type': 'pick', 'sku': sku, 'quantity': demand, 'slot': details['slot']})
                else:
                    logger.warning(f"[WARNING] Slot '{details['slot']}' coordinates not found for SKU '{sku}'")
            else:
                logger.warning(f"[WARNING] SKU '{sku}' not found in slotting assignments")

        if len(node_locations) <= 1: logger.error(
            "[ERROR] Not enough valid pick locations to create routes"); return None

        data = {'locations': node_locations, 'num_vehicles': self.picker_config['num_pickers'], 'depot': 0,
                'demands': demands,
                'vehicle_capacities': [self.picker_config['picker_capacity']] * self.picker_config['num_pickers']}
        manager = pywrapcp.RoutingIndexManager(len(data['locations']), data['num_vehicles'], data['depot'])
        routing = pywrapcp.RoutingModel(manager)

        def time_callback(from_index: int, to_index: int) -> int:
            from_node, to_node = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
            travel_time = calculate_manhattan_distance(data['locations'][from_node], data['locations'][to_node]) / \
                          self.picker_config['picker_speed']
            total_time = travel_time + service_times[to_node]
            return int(total_time * self.time_precision_scaler)

        transit_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        demand_callback_index = routing.RegisterUnaryTransitCallback(lambda i: data['demands'][manager.IndexToNode(i)])
        routing.AddDimensionWithVehicleCapacity(demand_callback_index, 0, data['vehicle_capacities'], True, 'Capacity')
        routing.AddDimension(transit_callback_index, 0,
                             int(self.picker_config['max_tour_time'] * self.time_precision_scaler), True, 'Time')

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = getattr(routing_enums_pb2.FirstSolutionStrategy,
                                                            self.solver_config['first_solution_strategy'])
        search_parameters.local_search_metaheuristic = getattr(routing_enums_pb2.LocalSearchMetaheuristic,
                                                               self.solver_config['local_search_metaheuristic'])
        search_parameters.time_limit.FromSeconds(self.solver_config['time_limit_seconds'])

        logger.info("Solving VRP...");
        solution = routing.SolveWithParameters(search_parameters)
        if solution:
            logger.info("[SUCCESS] VRP Solution Found!")
            return self._process_solution(solution, manager, routing, data, node_details_list)
        else:
            logger.error("[ERROR] No VRP solution found");
            return None

    def _process_solution(self, solution, manager, routing, data, node_details_list) -> Dict[str, Any]:
        time_dimension = routing.GetDimensionOrDie('Time')
        route_stats = {'route_times': [], 'route_loads': [], 'route_distances': []}
        summary_csv_data = []  # MODIFIED: List for the new summary CSV

        logger.info("\n" + "=" * 60 + "\nOPTIMIZED PICKER ROUTES\n" + "=" * 60)
        for vehicle_id in range(data['num_vehicles']):
            index, route_nodes, route_load, route_distance = routing.Start(vehicle_id), [], 0, 0
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_nodes.append(node_index)
                route_load += data['demands'][node_index]
                if not routing.IsEnd(solution.Value(routing.NextVar(index))):
                    next_node_index = manager.IndexToNode(solution.Value(routing.NextVar(index)))
                    route_distance += calculate_manhattan_distance(data['locations'][node_index],
                                                                   data['locations'][next_node_index])
                index = solution.Value(routing.NextVar(index))

            route_time = solution.Value(time_dimension.CumulVar(routing.End(vehicle_id))) / self.time_precision_scaler

            if len(route_nodes) > 1:
                route_stats['route_times'].append(route_time)
                route_stats['route_loads'].append(route_load)
                route_stats['route_distances'].append(route_distance)

                path_str_list = ["DISPATCH_ZONE"]
                for node_index in route_nodes:
                    details = node_details_list[node_index]
                    if details['type'] == 'pick':
                        path_str_list.append(f"{details['sku']}[{details['quantity']}u]")
                path_str_list.append("DISPATCH_ZONE")

                logger.info(
                    f"\nPicker {vehicle_id + 1} Route:\n  Path: {' -> '.join(path_str_list)}\n  Time: {route_time:.2f} min | Load: {route_load} units | Distance: {route_distance:.1f} m")

                # MODIFIED: Append one summary row per picker to the CSV data list
                summary_csv_data.append({
                    'picker_id': vehicle_id + 1,
                    'total_time_min': round(route_time, 2),
                    'total_quantity_picked': route_load,
                    'total_distance_m': round(route_distance, 1)
                })
            else:
                route_stats['route_times'].append(0)
                route_stats['route_loads'].append(0)
                route_stats['route_distances'].append(0)

        # MODIFIED: Save the new summary CSV file
        if summary_csv_data:
            summary_df = pd.DataFrame(summary_csv_data)
            summary_df.to_csv(ROUTE_SUMMARY_FILE, index=False)
            logger.info(f"\n[SUCCESS] Picker route summary saved to '{ROUTE_SUMMARY_FILE}'")

        return {'route_stats': route_stats,
                'solver_data': {'manager': manager, 'routing': routing, 'data': data, 'solution': solution}}


class RouteVisualizer:
    def __init__(self, warehouse_params: Dict[str, Any]):
        self.params = warehouse_params
        self.total_width = (warehouse_params['racks'] / 2) * (
                    2 * warehouse_params['rack_depth'] + warehouse_params['picking_aisle_width'])
        self.total_length = warehouse_params['bays'] * warehouse_params['bay_width'] + warehouse_params[
            'cross_aisle_width']

    def visualize_routes(self, solution, manager, routing, data, route_stats):
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        self._plot_warehouse_layout(ax1)
        self._plot_routes(ax1, solution, manager, routing, data)
        self._plot_route_statistics(ax2, route_stats)
        plt.tight_layout()
        plt.savefig('picker_routes_analysis.png', dpi=300)
        plt.show()

    def _plot_warehouse_layout(self, ax):
        for r_pair in range(self.params['racks'] // 2):
            x_base = r_pair * (2 * self.params['rack_depth'] + self.params['picking_aisle_width'])
            ax.add_patch(patches.Rectangle((x_base, 0), self.params['rack_depth'],
                                           self.params['bays'] * self.params['bay_width'], facecolor='lightgray',
                                           alpha=0.7))
            ax.add_patch(patches.Rectangle((x_base + self.params['rack_depth'] + self.params['picking_aisle_width'], 0),
                                           self.params['rack_depth'], self.params['bays'] * self.params['bay_width'],
                                           facecolor='lightgray', alpha=0.7))
        ax.set_xlim(-2, self.total_width + 2);
        ax.set_ylim(-5, self.total_length + 5);
        ax.set_title('Warehouse Layout & Optimized Routes');
        ax.grid(True, alpha=0.3)

    def _plot_routes(self, ax, solution, manager, routing, data):
        colors, locations = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12'], data['locations']
        if len(locations) > 1: ax.scatter([loc[0] for loc in locations[1:]], [loc[1] for loc in locations[1:]],
                                          c='darkblue', s=60, label='Pick Locations', zorder=5)
        ax.scatter(locations[0][0], locations[0][1], c='red', marker='s', s=200, label='Dispatch Zone', zorder=6)
        for v_id in range(data['num_vehicles']):
            index, route_nodes = routing.Start(v_id), []
            while not routing.IsEnd(index): route_nodes.append(manager.IndexToNode(index)); index = solution.Value(
                routing.NextVar(index))
            route_nodes.append(manager.IndexToNode(index))
            if len(route_nodes) > 2: ax.plot([locations[n][0] for n in route_nodes],
                                             [locations[n][1] for n in route_nodes], color=colors[v_id % len(colors)],
                                             lw=3, marker='o', label=f'Picker {v_id + 1}')
        ax.legend()

    def _plot_route_statistics(self, ax, stats):
        names, x, width = [f"Picker {i + 1}" for i in range(len(stats['route_times']))], np.arange(
            len(stats['route_times'])), 0.25
        ax.bar(x - width, stats['route_times'], width, label='Time (min)', color='#3498db')
        ax.bar(x, stats['route_loads'], width, label='Items', color='#2ecc71')
        ax.bar(x + width, [d / 10 for d in stats['route_distances']], width, label='Distance (10m)', color='#e74c3c')
        ax.set_ylabel('Values');
        ax.set_title('Route Performance');
        ax.set_xticks(x);
        ax.set_xticklabels(names);
        ax.legend();
        ax.grid(True, alpha=0.3)


def main():
    """Main function to orchestrate the VRP routing optimization."""
    try:
        logger.info("Starting Warehouse VRP Routing System")
        if not (validate_warehouse_config(WAREHOUSE_PARAMS) and validate_picker_config(PICKER_CONFIG)): return

        sku_to_details = SlottingDataLoader(SLOTTING_RESULTS_FILE).load_assignments()
        if not sku_to_details: return

        coordinate_system = WarehouseCoordinateSystem(WAREHOUSE_PARAMS, DISPATCH_ZONE_LOCATION)
        order_data = load_and_consolidate_orders(ORDER_FILE_PATH)
        if not order_data: return

        results = VRPSolver(PICKER_CONFIG, SOLVER_CONFIG).solve_picker_vrp(
            order_data[0], order_data[1], sku_to_details,
            coordinate_system.slot_coords, coordinate_system.dispatch_coords
        )

        if results:
            RouteVisualizer(WAREHOUSE_PARAMS).visualize_routes(
                results['solver_data']['solution'], results['solver_data']['manager'],
                results['solver_data']['routing'], results['solver_data']['data'],
                results['route_stats']
            )
            logger.info("\n[SUCCESS] VRP optimization completed successfully!")
        else:
            logger.error("[ERROR] VRP optimization failed")

    except Exception as e:
        logger.error(f"[ERROR] Critical error in VRP routing: {e}", exc_info=True)
    finally:
        logger.info("VRP Routing System Complete")


if __name__ == "__main__":
    main()