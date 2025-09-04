import pandas as pd
import numpy as np
from collections import defaultdict
import math
import logging
import time
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from statsmodels.tsa.arima.model import ARIMA
import gurobipy as gp
from gurobipy import GRB

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('warehouse_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- 1. CONFIGURATION ---
# Default configuration - can be overridden by config.yaml
DEFAULT_CONFIG = {
    'file_settings': {
        # --- NOTE: Please change this to the path of your actual data file ---
        'file_path': 'Sales.XLSX',
        'order_id_column': 'Sales Order No.',
        'sku_column': 'Material Number',
        'quantity_column': 'Invoiced Qty',
        'date_column': 'Billing Date'
    },
    'warehouse': {
        'racks': 30,
        'bays': 24,
        'levels': 7,
        'middle_aisle_after_bay': 11,
        'bay_width': 1.0,
        'rack_depth': 1.0,
        'picking_aisle_width': 3.0,
        'cross_aisle_width': 4.0,
        'dispatch_zone_location': 'front'
    },
    'optimization': {
        'mode': 'forecast',
        'distance_metric': 'manhattan',  # 'manhattan' is recommended and the only supported metric
        'weights': {
            'forecast': 1.0
        },
        'gurobi_params': {
            'TimeLimit': 300,
            'MIPFocus': 1,
            'MIPGap': 0.01
        }
    }
}


def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """Load configuration from YAML file or use defaults."""
    config_file = Path(config_path)
    config = DEFAULT_CONFIG.copy()  # Start with defaults
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                user_config = yaml.safe_load(f)
            # Deep merge user config into defaults
            for section, values in user_config.items():
                if section in config and isinstance(config[section], dict):
                    config[section].update(values)
                else:
                    config[section] = values
            logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}. Using defaults.")
    else:
        logger.info("config.yaml not found. Using default configuration.")
    return config


def validate_warehouse_config(params: Dict[str, Any]) -> bool:
    """Validate warehouse configuration parameters."""
    try:
        if params['racks'] <= 0 or params['bays'] <= 0 or params['levels'] <= 0:
            raise ValueError("Racks, bays, and levels must be positive")

        if params['middle_aisle_after_bay'] >= params['bays']:
            raise ValueError("Middle aisle position must be less than total bays")

        if any(params[key] <= 0 for key in ['bay_width', 'rack_depth', 'picking_aisle_width', 'cross_aisle_width']):
            raise ValueError("All width/depth parameters must be positive")

        logger.info("✅ Warehouse configuration validated successfully")
        return True

    except (ValueError, KeyError) as e:
        logger.error(f"❌ Warehouse configuration validation failed: {e}")
        return False


class DataLoader:
    """Handles loading and initial cleaning of the source data."""

    def __init__(self, file_path: str, column_map: Dict[str, str]):
        self.file_path = file_path
        self.column_map = column_map

    def validate_order_data(self, df: pd.DataFrame) -> None:
        """Validate order data quality."""
        logger.info("Performing data quality checks...")

        # Check for negative quantities
        negative_qty = (df['quantity'] < 0).sum()
        if negative_qty > 0:
            logger.warning(f"⚠️ {negative_qty} negative quantities found and will be filtered")

        # Check for future dates (assuming historical data)
        future_dates = (df['date'] > pd.Timestamp.now()).sum()
        if future_dates > 0:
            logger.warning(f"⚠️ {future_dates} future dates found in data")

        # Check for duplicate order lines
        duplicates = df.duplicated(['order_id', 'sku']).sum()
        if duplicates > 0:
            logger.warning(f"⚠️ {duplicates} duplicate order lines found")

        # Check data distribution
        logger.info(f"Data span: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"Unique SKUs: {df['sku'].nunique()}")
        logger.info(f"Total orders: {df['order_id'].nunique()}")

    def load_and_clean(self) -> Optional[pd.DataFrame]:
        """Load and clean order data with comprehensive error handling."""
        logger.info("Step 1: Loading and cleaning data...")

        try:
            # Load data based on file type
            if self.file_path.lower().endswith('.csv'):
                df = pd.read_csv(self.file_path, engine='python', on_bad_lines='warn', encoding='latin1')
            elif self.file_path.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(self.file_path, engine='openpyxl')
            else:
                raise ValueError(f"Unsupported file format: {self.file_path}")

            # Clean column names
            df.columns = df.columns.str.strip()

            # Validate required columns exist
            missing_columns = set(self.column_map.keys()) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Rename columns to standard names
            df.rename(columns=self.column_map, inplace=True)

            # Select and clean required columns
            order_data = df[['order_id', 'sku', 'quantity', 'date']].copy()

            # Data type conversions with error handling
            order_data['date'] = pd.to_datetime(order_data['date'], errors='coerce')
            order_data['sku'] = order_data['sku'].astype(str)
            order_data['order_id'] = order_data['order_id'].astype(str)
            order_data['quantity'] = pd.to_numeric(order_data['quantity'], errors='coerce').fillna(0)

            # Remove invalid records
            initial_count = len(order_data)
            order_data.dropna(subset=['date', 'order_id', 'sku'], inplace=True)
            order_data = order_data[order_data['quantity'] > 0]

            final_count = len(order_data)
            if initial_count > final_count:
                logger.warning(f"Filtered out {initial_count - final_count} invalid records")

            # Validate data quality
            self.validate_order_data(order_data)

            logger.info(f"✅ Successfully loaded {final_count} valid order lines")
            return order_data

        except FileNotFoundError:
            logger.error(f"❌ File not found: '{self.file_path}'. Please check the path in your config.")
            return None
        except Exception as e:
            logger.error(f"❌ Error during data loading: {e}")
            return None


class Forecaster:
    """Generates demand priority scores for each SKU based on historical data."""

    def __init__(self, order_data: pd.DataFrame):
        self.order_data = order_data

    def run_sku_prioritization(self) -> pd.DataFrame:
        """Generate priority scores using historical demand, frequency, and stability."""
        logger.info("Step 2: Running SKU prioritization based on historical data...")

        # Calculate basic statistics
        sku_stats = self.order_data.groupby('sku')['quantity'].agg([
            'sum', 'std', 'count', 'mean'
        ]).reset_index()

        sku_stats.rename(columns={
            'sum': 'total_demand',
            'std': 'volatility',
            'count': 'order_frequency',
            'mean': 'avg_order_size'
        }, inplace=True)

        # Fill NaN volatility with 0 for single-order SKUs
        sku_stats['volatility'] = sku_stats['volatility'].fillna(0)

        # Calculate demand coefficient of variation
        sku_stats['demand_coefficient_of_variation'] = (
                sku_stats['volatility'] / sku_stats['avg_order_size']
        ).fillna(0)

        def normalize_series(series: pd.Series) -> pd.Series:
            """Normalize series to 0-1 range."""
            min_val, max_val = series.min(), series.max()
            if max_val > min_val:
                return (series - min_val) / (max_val - min_val)
            return pd.Series([0.5] * len(series), index=series.index)  # Return 0.5 if all values are same

        # Normalize metrics for scoring
        sku_stats['norm_demand'] = normalize_series(sku_stats['total_demand'])
        sku_stats['norm_frequency'] = normalize_series(sku_stats['order_frequency'])
        sku_stats['norm_stability'] = 1 - normalize_series(sku_stats['demand_coefficient_of_variation'])

        # Calculate composite priority score (higher score = higher priority)
        sku_stats['priority_score'] = (
                0.5 * sku_stats['norm_demand'] +
                0.3 * sku_stats['norm_frequency'] +
                0.2 * sku_stats['norm_stability']
        )

        logger.info(f"✅ Prioritization complete for {len(sku_stats)} SKUs")
        return sku_stats


class WarehouseLayout:
    """Creates and manages the warehouse grid based on a realistic rack-and-aisle layout."""

    def __init__(self, layout_params: Dict[str, Any]):
        self.params = layout_params
        self.racks = self.params['racks']
        self.bays = self.params['bays']
        self.levels = self.params['levels']
        self.middle_aisle_position = self.params['middle_aisle_after_bay']
        self.bay_width = self.params['bay_width']
        self.rack_depth = self.params['rack_depth']
        self.picking_aisle_width = self.params['picking_aisle_width']
        self.cross_aisle_width = self.params['cross_aisle_width']

        # Calculate warehouse dimensions
        self.total_width = (self.racks / 2) * (2 * self.rack_depth + self.picking_aisle_width)
        self.total_length = self.bays * self.bay_width + self.cross_aisle_width

        # Create layout and dispatch zone
        self.layout_df = self._create_layout()
        self.dispatch_zone = self._get_dispatch_zone_coords(self.params['dispatch_zone_location'])

    def _get_slot_coords(self, rack: int, bay: int) -> Tuple[float, float]:
        """Calculate physical coordinates for a rack-bay position."""
        # Determine which rack pair and position within pair
        rack_pair_index = (rack - 1) // 2
        x = rack_pair_index * (2 * self.rack_depth + self.picking_aisle_width)

        # Adjust for second rack in pair
        if (rack - 1) % 2 == 1:
            x += self.rack_depth + self.picking_aisle_width

        # Calculate y position with middle aisle consideration
        y = (bay - 1) * self.bay_width
        if bay > self.middle_aisle_position:
            y += self.cross_aisle_width

        return (x, y)

    def _create_layout(self) -> pd.DataFrame:
        """Create comprehensive warehouse layout dataframe."""
        logger.info("Step 3: Creating realistic warehouse layout...")

        slots = []
        for r in range(1, self.racks + 1):
            for b in range(1, self.bays + 1):
                coords = self._get_slot_coords(r, b)
                for l in range(1, self.levels + 1):
                    slots.append({
                        'slot_id': f"R{r:02d}-B{b:02d}-L{l:02d}",
                        'rack': r,
                        'bay': b,
                        'level': l,
                        'x_coord': coords[0],
                        'y_coord': coords[1]
                    })

        layout_df = pd.DataFrame(slots)

        logger.info(f"✅ Created warehouse with {self.racks} racks, {self.bays} bays, {self.levels} levels")
        logger.info(f"Total storage slots: {len(layout_df)}")

        return layout_df

    def _get_dispatch_zone_coords(self, location: str) -> Tuple[float, float]:
        """Calculate dispatch zone coordinates based on location preference."""
        if location == 'front':
            return (self.total_width / 2, self.total_length + self.cross_aisle_width / 2)
        elif location == 'back':
            return (self.total_width / 2, -self.cross_aisle_width / 2)
        elif location == 'center':
            return (self.total_width / 2, self.total_length / 2)
        else:
            logger.warning(f"Unknown dispatch zone location '{location}', using 'front'")
            return (self.total_width / 2, self.total_length + self.cross_aisle_width / 2)


class SlottingOptimizer:
    """Configures and runs the Gurobi optimization model."""

    def __init__(self, skus: List[str], layout: pd.DataFrame, sku_data: pd.DataFrame, config: Dict[str, Any]):
        self.skus = skus
        self.layout = layout
        self.sku_data = sku_data
        self.config = config
        self.model = None
        self._setup_model()

    def _setup_model(self) -> None:
        """Initialize Gurobi model."""
        try:
            self.model = gp.Model("WarehouseSlottingMILP")
            self.model.setParam('OutputFlag', 1)
            self.model.setParam('LogFile', 'gurobi.log')
            logger.info("✅ Gurobi model initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gurobi model: {e}")
            raise

    def _add_distance_constraints(self, dist_var, dx, dy, name_suffix: str) -> None:
        """Add distance calculation constraints. Only Manhattan is supported."""
        metric = self.config.get('distance_metric', 'manhattan')
        if metric != 'manhattan':
            logger.warning(f"Unsupported distance_metric '{metric}' found. Defaulting to 'manhattan'.")

        abs_dx = self.model.addVar(name=f"abs_dx_{name_suffix}")
        abs_dy = self.model.addVar(name=f"abs_dy_{name_suffix}")

        self.model.addConstr(abs_dx >= dx, name=f"abs_dx_pos_{name_suffix}")
        self.model.addConstr(abs_dx >= -dx, name=f"abs_dx_neg_{name_suffix}")
        self.model.addConstr(abs_dy >= dy, name=f"abs_dy_pos_{name_suffix}")
        self.model.addConstr(abs_dy >= -dy, name=f"abs_dy_neg_{name_suffix}")
        self.model.addConstr(dist_var == abs_dx + abs_dy, name=f"manhattan_dist_{name_suffix}")

    def _progress_callback(self, model, where) -> None:
        """Callback function to report optimization progress."""
        if where == gp.GRB.Callback.MIP:
            objbst = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
            objbnd = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
            time_elapsed = model.cbGet(gp.GRB.Callback.RUNTIME)

            if objbst < 1e30 and abs(objbst) > 1e-6:
                gap = abs((objbst - objbnd) / objbst) * 100
                logger.info(
                    f"Progress: Best={objbst:.2f}, Bound={objbnd:.2f}, Gap={gap:.2f}%, Time={time_elapsed:.1f}s")

    def run_milp_optimization(self) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """Run the MILP optimization with comprehensive error handling."""
        logger.info("Step 4: Running Gurobi MILP Optimization...")

        try:
            slot_ids = self.layout['slot_id'].tolist()
            # Decision variables: x[slot_id, sku] = 1 if SKU assigned to slot
            x = self.model.addVars(slot_ids, self.skus, vtype=GRB.BINARY, name="assign")

            # --- Objective function ---
            obj = gp.LinExpr()
            sku_priority_map = self.sku_data.set_index('sku')['priority_score']

            if self.config['mode'] == 'forecast':
                logger.info("  Building priority-based objective function...")
                dist_to_dispatch_vars = self.model.addVars(self.skus, name="dist_dispatch")

                # Get SKU coordinates as weighted averages
                sku_coords_x = {sku: gp.quicksum(
                    x[s_id, sku] * row['x_coord'] for s_id, row in self.layout.set_index('slot_id').iterrows()) for sku
                                in self.skus}
                sku_coords_y = {sku: gp.quicksum(
                    x[s_id, sku] * row['y_coord'] for s_id, row in self.layout.set_index('slot_id').iterrows()) for sku
                                in self.skus}

                for sku in self.skus:
                    # Calculate distance from the SKU's assigned location to the dispatch zone
                    dx_d = sku_coords_x[sku] - self.config['dispatch_zone'][0]
                    dy_d = sku_coords_y[sku] - self.config['dispatch_zone'][1]
                    self._add_distance_constraints(dist_to_dispatch_vars[sku], dx_d, dy_d, f"dispatch_{sku}")

                    # Objective: Minimize sum of (priority_score * distance)
                    priority_score = sku_priority_map.get(sku, 0)
                    obj += self.config['weights']['forecast'] * priority_score * dist_to_dispatch_vars[sku]

            self.model.setObjective(obj, GRB.MINIMIZE)

            # --- Constraints ---
            # Each SKU must be assigned to exactly one slot
            for sku in self.skus:
                self.model.addConstr(x.sum('*', sku) == 1, name=f"SKU_placement_{sku}")

            # Each slot can have at most one SKU
            for slot_id in slot_ids:
                self.model.addConstr(x.sum(slot_id, '*') <= 1, name=f"Slot_capacity_{slot_id}")

            # Set Gurobi parameters from config
            if 'gurobi_params' in self.config:
                for param, value in self.config['gurobi_params'].items():
                    self.model.setParam(param, value)
                    logger.info(f"  Set Gurobi parameter: {param} = {value}")

            # Solve with callback
            logger.info("  Starting optimization...")
            start_time = time.time()
            self.model.optimize(self._progress_callback)
            solve_time = time.time() - start_time

            # Process results
            assignments = {}
            performance = {'solve_time': solve_time, 'status': self.model.status,
                           'status_description': self._get_status_description()}

            if self.model.status == GRB.OPTIMAL:
                logger.info("✅ Optimal solution found!")
                performance.update({'objective_value': self.model.objVal, 'optimality_gap': '0.00%',
                                    'nodes_explored': self.model.NodeCount})
            elif self.model.status == GRB.TIME_LIMIT and self.model.solCount > 0:
                logger.info("⏱️ Time limit reached, but feasible solution found")
                performance.update(
                    {'objective_value': self.model.objVal, 'optimality_gap': f"{self.model.MIPGap * 100:.2f}%",
                     'nodes_explored': self.model.NodeCount})
            else:
                logger.error(f"❌ No feasible solution found. Status: {self._get_status_description()}")
                if self.model.status == GRB.INFEASIBLE:
                    logger.info("Computing IIS to identify conflicting constraints...")
                    self.model.computeIIS()
                    self.model.write("model_iis.ilp")
                    logger.info("IIS written to model_iis.ilp")
                return assignments, performance

            # Extract solution
            if self.model.solCount > 0:
                solution = self.model.getAttr('X', x)
                for slot_id, sku in solution.keys():
                    if solution[slot_id, sku] > 0.5:
                        assignments[sku] = slot_id

            return assignments, performance

        except gp.GurobiError as e:
            logger.error(f"❌ Gurobi error: {e}")
            return {}, {'error': str(e)}
        except Exception as e:
            logger.error(f"❌ Unexpected error in optimization: {e}")
            return {}, {'error': str(e)}

    def _get_status_description(self) -> str:
        """Get human-readable description of optimization status."""
        status_map = {
            GRB.OPTIMAL: "Optimal solution found", GRB.INFEASIBLE: "Model is infeasible",
            GRB.INF_OR_UNBD: "Model is infeasible or unbounded", GRB.UNBOUNDED: "Model is unbounded",
            GRB.CUTOFF: "Objective cutoff reached", GRB.ITERATION_LIMIT: "Iteration limit reached",
            GRB.NODE_LIMIT: "Node limit reached", GRB.TIME_LIMIT: "Time limit reached",
            GRB.SOLUTION_LIMIT: "Solution limit reached", GRB.INTERRUPTED: "Optimization interrupted",
            GRB.NUMERIC: "Numerical issues encountered", GRB.SUBOPTIMAL: "Suboptimal solution found"
        }
        return status_map.get(self.model.status, f"Unknown status: {self.model.status}")


class Validator:
    """Performs comprehensive validation on the optimization output."""

    def __init__(self, assignments: Dict[str, str], skus_to_place: List[str]):
        self.assignments = assignments
        self.skus_to_place = skus_to_place

    def validate(self) -> bool:
        """Perform comprehensive validation of the solution."""
        logger.info("Step 5: Validating optimization output...")
        is_valid = True

        # Check for unassigned SKUs
        unassigned_skus = set(self.skus_to_place) - set(self.assignments.keys())
        if unassigned_skus:
            logger.error(f"❌ Validation Error: {len(unassigned_skus)} unassigned SKUs")
            if len(unassigned_skus) <= 10:
                logger.error(f"Unassigned SKUs: {list(unassigned_skus)}")
            is_valid = False

        # Check for duplicate slot assignments
        assigned_slots = list(self.assignments.values())
        if len(assigned_slots) != len(set(assigned_slots)):
            duplicates = len(assigned_slots) - len(set(assigned_slots))
            logger.error(f"❌ Validation Error: {duplicates} duplicate slot assignments")
            is_valid = False

        if is_valid:
            logger.info("✅ Validation Successful: Solution is valid")
            logger.info(f"Successfully assigned {len(self.assignments)} SKUs to unique slots")
        return is_valid


def main():
    """Main function to orchestrate the slotting optimization process."""
    start_time = time.time()
    try:
        logger.info("🏭 Starting Warehouse Slotting Optimization System")

        # Load and validate configuration
        config = load_config()
        if not validate_warehouse_config(config['warehouse']):
            return

        # Prepare column mapping
        file_config = config['file_settings']
        column_map = {
            file_config['order_id_column']: 'order_id', file_config['sku_column']: 'sku',
            file_config['quantity_column']: 'quantity', file_config['date_column']: 'date'
        }

        # Step 1: Load and clean data
        data_loader = DataLoader(file_config['file_path'], column_map)
        order_data = data_loader.load_and_clean()
        if order_data is None: return

        # Step 2: Generate SKU priority scores
        forecaster = Forecaster(order_data)
        sku_priority_data = forecaster.run_sku_prioritization()

        # Step 3: Create warehouse layout
        warehouse = WarehouseLayout(config['warehouse'])

        # Prepare the final optimizer configuration
        optimizer_config = config['optimization'].copy()
        optimizer_config['dispatch_zone'] = warehouse.dispatch_zone

        # Step 4: Determine SKUs to place
        all_skus = order_data['sku'].unique().tolist()
        num_slots = len(warehouse.layout_df)
        if num_slots < len(all_skus):
            logger.warning(f"⚠️ Limited slots ({num_slots}) for {len(all_skus)} SKUs. Prioritizing by score.")
            skus_to_place = sku_priority_data.nlargest(num_slots, 'priority_score')['sku'].tolist()
        else:
            skus_to_place = all_skus
        logger.info(f"📦 Optimizing placement for {len(skus_to_place)} SKUs")

        # Step 5: Run optimization
        optimizer = SlottingOptimizer(skus_to_place, warehouse.layout_df, sku_priority_data, optimizer_config)
        final_assignments, performance_log = optimizer.run_milp_optimization()

        # Step 6: Validate and save results
        if final_assignments and Validator(final_assignments, skus_to_place).validate():
            results_df = pd.DataFrame(list(final_assignments.items()), columns=['SKU', 'Assigned_Slot'])
            results_df = results_df.merge(warehouse.layout_df, left_on='Assigned_Slot', right_on='slot_id', how='left')
            results_df = results_df.merge(
                sku_priority_data, left_on='SKU', right_on='sku', how='left'
            ).drop(columns=['sku', 'slot_id'])

            # Calculate distance to dispatch zone for analysis
            dispatch_x, dispatch_y = warehouse.dispatch_zone
            results_df['distance_to_dispatch'] = abs(results_df['x_coord'] - dispatch_x) + abs(
                results_df['y_coord'] - dispatch_y)

            # Select and order final columns
            final_columns = [
                'SKU', 'Assigned_Slot', 'rack', 'bay', 'level', 'priority_score', 'total_demand',
                'volatility', 'order_frequency', 'distance_to_dispatch', 'x_coord', 'y_coord'
            ]
            results_df = results_df[final_columns].sort_values(by='priority_score', ascending=False).reset_index(
                drop=True)

            # Save results
            output_filename = 'slotting_assignments_output.csv'
            results_df.to_csv(output_filename, index=False)
            logger.info(f"\n✅ Results saved to '{output_filename}'")

            # Display performance metrics and summary
            logger.info("\n" + "=" * 50 + "\n📊 OPTIMIZATION PERFORMANCE\n" + "=" * 50)
            for key, value in performance_log.items():
                if key != 'error': logger.info(f"  📈 {key.replace('_', ' ').title()}: {value}")

            logger.info("\n" + "=" * 50 + "\n📋 ASSIGNMENT SUMMARY\n" + "=" * 50)
            high_priority = results_df[results_df['priority_score'] >= 0.8]
            logger.info(f"  🔥 High Priority SKUs (score ≥ 0.8): {len(high_priority)}")
            logger.info(f"  📦 Total SKUs Assigned: {len(results_df)}")
            logger.info(f"  🎯 Average Distance to Dispatch: {results_df['distance_to_dispatch'].mean():.2f}m")

            # Display top 10 assignments
            logger.info("\n" + "=" * 50 + "\n🏆 TOP 10 PRIORITY ASSIGNMENTS\n" + "=" * 50)
            top_10 = results_df.head(10)[['SKU', 'Assigned_Slot', 'priority_score', 'distance_to_dispatch']]
            for _, row in top_10.iterrows():
                logger.info(
                    f"  - 📦 {row['SKU']:<15} → {row['Assigned_Slot']:<12} (Score: {row['priority_score']:.3f}, Dist: {row['distance_to_dispatch']:.1f}m)")

        else:
            logger.error("❌ Optimization completed without a valid solution.")
            if 'error' in performance_log:
                logger.error(f"Error details: {performance_log['error']}")

    except Exception as e:
        logger.error(f"❌ Critical error in main execution: {e}", exc_info=True)
    finally:
        total_time = time.time() - start_time
        logger.info(f"\n⏱️ Total execution time: {total_time:.2f} seconds")
        logger.info("🏁 Warehouse Slotting Optimization Complete")


if __name__ == '__main__':
    main()