import pandas as pd
import numpy as np
import simpy
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde # Ensure this import is present at the top of your file


class Station:
    def __init__(self, env):
        self.env = env
        self.queue = []  # store arrival times
        self.waiting_times = []  # store computed waiting times

    def arrive(self, _: float):
        """A passenger arrives at current env time and joins the queue."""
        self.queue.append(self.env.now)

    def board_train(self, departure_time: float):
        """Board all waiting passengers at a train departure and record their waiting times."""
        boarded = []
        for arrival_time in self.queue:
            if arrival_time <= departure_time:
                self.waiting_times.append(departure_time - arrival_time)
                boarded.append(arrival_time)
        # remove boarded passengers from queue
        self.queue = [t for t in self.queue if t not in boarded]


def load_excel_data(path: str):
    """
    Load busyness and inbound timetables from an Excel file.
    Sheet1 (index 0): 'Time' (datetime.time or string like '03:45:00 PM'), 'Sunday', 'Monday-Thursday Average'
    Sheet2 (index 1): 'Sunday Inbound', 'Monday-Thursday Inbound'

    Returns:
      demand: dict of DataFrames with columns ['hour', 'busyness'] per scenario key
      timetables: dict of lists of departure times (seconds since midnight) per scenario key
    """
    df_busyness = pd.read_excel(path, sheet_name=0)
    df_tt = pd.read_excel(path, sheet_name=1)

    # Parse and normalize Time column for busyness
    times_raw = df_busyness['Time']
    times_dt = pd.to_datetime(times_raw.astype(str), format='%H:%M:%S', errors='coerce')
    times_sec = times_dt.dt.hour * 3600 + times_dt.dt.minute * 60 + times_dt.dt.second

    demand = {}
    mapping = [('Sunday', 'Sunday'), ('Monday-Thursday Average', 'MonThu')]
    for col, key in mapping:
        df = pd.DataFrame({
            'hour': (times_sec // 3600).astype(int),
            'busyness': df_busyness[col].astype(float)
        })
        demand[key] = df

    timetables = {}
    mapping_tt = [('Sunday Inbound', 'Sunday'), ('Monday-Thursday Inbound', 'MonThu')]
    for col, key in mapping_tt:
        times_raw_tt = df_tt[col].dropna()
        secs = []
        for t in times_raw_tt:
            dt = pd.to_datetime(str(t), errors='coerce')
            if pd.isna(dt):
                dt = pd.to_datetime(str(t), format='%I:%M:%S %p', errors='coerce')
            secs.append(dt.hour * 3600 + dt.minute * 60 + dt.second)
        timetables[key] = sorted(secs)

    return demand, timetables


def generate_even_timetable(original_timetable: list[float], start_sec: float = 0, end_sec: float = 24*3600) -> list[float]:
    """
    Redistribute the same number of trains evenly between start_sec and end_sec.
    """
    count = len(original_timetable)
    if count == 0:
        return []
    interval = (end_sec - start_sec) / count
    return [start_sec + i * interval for i in range(count)]


def generate_weighted_timetable(demand_df: pd.DataFrame, original_timetable: list[float],
                                start_hour: int = 0, end_hour: int = 24) -> list[float]:
    """
    Redistribute trains by sampling departure times based on demand busyness weights.
    Only times between start_hour and end_hour (inclusive start, exclusive end) are considered.
    """
    df_period = demand_df[(demand_df['hour'] >= start_hour) & (demand_df['hour'] < end_hour)].copy()
    if df_period.empty:
        return []
    weights = df_period['busyness'].values
    weights = weights / weights.sum()
    count = len(original_timetable)
    hours = np.random.choice(df_period['hour'], size=count, p=weights)
    secs = []
    for h in hours:
        secs.append(h * 3600 + np.random.uniform(0, 3600))
    return sorted(secs)

def generate_minmax_timetable(original_timetable: list[float]) -> list[float]:
    """
    Redistribute trains evenly between the first and last existing departures,
    minimizing the maximum headway and thus the maximum possible wait.
    """
    if not original_timetable:
        return []
    start = original_timetable[0]
    end = original_timetable[-1]
    return generate_even_timetable(original_timetable, start, end)

def passenger_generator(env: simpy.Environment, station: Station,
                        demand_df: pd.DataFrame, total_passengers: int):
    """
    Generate passenger arrivals based on demand weights.
    """
    weights = demand_df['busyness'] / demand_df['busyness'].sum()
    for idx, row in demand_df.iterrows():
        hour = row['hour']
        n_arrivals = int(np.round(weights[idx] * total_passengers))
        start = hour * 3600
        end = (hour + 1) * 3600
        arrival_times = np.random.uniform(start, end, size=n_arrivals)
        for at in np.sort(arrival_times):
            yield env.timeout(at - env.now)
            station.arrive(at)


def train_process(env: simpy.Environment, station: Station,
                  timetable_sec: list[float]):
    """
    Simulate train departures according to timetable_sec.
    """
    for departure in timetable_sec:
        yield env.timeout(departure - env.now)
        station.board_train(departure)


def run_simulation(excel_path: str):
    """
    Run simulations for real and generated timetable scenarios.
    Returns:
      demand: dict of demand DataFrames
      results: dict of waiting time lists by scenario key
    """
    demand, timetables = load_excel_data(excel_path)

    # Generate algorithmic timetables (same train count, redistributed)
    generated = {}
    for scenario, original in timetables.items():
        generated[f"{scenario}_even24"] = generate_even_timetable(original, 0, 24*3600)
        generated[f"{scenario}_even6_24"] = generate_even_timetable(original, 6*3600, 24*3600)
        generated[f"{scenario}_weighted"] = generate_weighted_timetable(demand[scenario], original, 0, 24)
        generated[f"{scenario}_minmax"] = generate_minmax_timetable(original)

    # Combine real and generated timetables
    all_timetables = {**timetables, **generated}

    results = {}
    for scenario, timetable in all_timetables.items():
        # Determine base demand key (strip suffix if generated)
        base_key = scenario.split('_')[0]
        total_percentage = demand[base_key]['busyness'].sum()
        total_passengers = int(1000 * total_percentage)

        env = simpy.Environment()
        station = Station(env)
        env.process(passenger_generator(env, station,
                                        demand[base_key], total_passengers))
        env.process(train_process(env, station, timetable))
        env.run(until=24 * 3600)

        results[scenario] = station.waiting_times

    return demand, results


def plot_demand(demand: dict[str, pd.DataFrame]):
    """Plot hourly busyness for each scenario."""
    plt.figure()
    for key, df in demand.items():
        hours = df['hour']
        busyness = df['busyness']
        plt.plot(hours, busyness, marker='o', label=key)
    plt.xlabel('Hour of Day')
    plt.ylabel('Busyness (relative)')
    plt.title('Hourly Busyness by Scenario')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_waiting_times(results: dict[str, list[float]]):
    """Plot waiting time distribution curves using KDE, comparing each baseline scenario
    to its variants, with average and max wait times in the legend."""
    max_minutes = 60
    # Create a smooth range of x values for plotting KDE
    x_grid = np.linspace(0, max_minutes, 500)

    potential_base_keys = sorted(list(set([key.split('_')[0] for key in results.keys()])))
    actual_base_keys = [bk for bk in potential_base_keys if bk in results]

    for base_key in actual_base_keys:
        plt.figure(figsize=(12, 7))

        # 1. Plot the baseline scenario itself
        if base_key in results:
            base_waits_raw = results[base_key]
            base_waits_min = np.array(base_waits_raw) / 60.0

            base_avg_wait_str = "N/A"
            base_max_wait_str = "N/A"

            if len(base_waits_min) > 1:  # KDE needs more than 1 point
                base_avg_wait = np.mean(base_waits_min)
                base_max_wait = np.max(base_waits_min)
                base_avg_wait_str = f"{base_avg_wait:.2f}"
                base_max_wait_str = f"{base_max_wait:.2f}"

                try:
                    kde = gaussian_kde(base_waits_min)
                    plt.plot(x_grid, kde(x_grid), linewidth=2.5,
                             label=f"{base_key} (Baseline, Avg: {base_avg_wait_str} min, Max: {base_max_wait_str} min)")
                except Exception as e:  # Catch potential errors with KDE on unusual data
                    print(f"Could not plot KDE for {base_key}: {e}")
                    # Fallback to an empty plot with label if KDE fails
                    plt.plot([], [], linewidth=2.5,
                             label=f"{base_key} (Baseline, Avg: {base_avg_wait_str} min, Max: {base_max_wait_str} min, KDE Error)")

            elif len(base_waits_min) == 1:  # Handle single point case (KDE not meaningful)
                base_avg_wait_str = f"{base_waits_min[0]:.2f}"
                base_max_wait_str = f"{base_waits_min[0]:.2f}"
                plt.plot([], [],
                         label=f"{base_key} (Baseline, Avg: {base_avg_wait_str} min, Max: {base_max_wait_str} min, Single Point)")
            else:  # No data
                plt.plot([], [], label=f"{base_key} (Baseline, Avg: N/A, Max: N/A, No Data)")

        # 2. Plot all variant scenarios related to this base_key
        variant_scenarios_data = []
        for scenario_key, scenario_waits_raw in results.items():
            if scenario_key.startswith(base_key + "_"):
                variant_scenarios_data.append((scenario_key, scenario_waits_raw))

        sorted_variants = sorted(variant_scenarios_data, key=lambda item: item[0])

        for variant_key, waits_raw in sorted_variants:
            waits_min = np.array(waits_raw) / 60.0

            avg_wait_str = "N/A"
            max_wait_str = "N/A"

            if len(waits_min) > 1:  # KDE needs more than 1 point
                avg_wait = np.mean(waits_min)
                max_wait = np.max(waits_min)
                avg_wait_str = f"{avg_wait:.2f}"
                max_wait_str = f"{max_wait:.2f}"

                try:
                    kde = gaussian_kde(waits_min)
                    plt.plot(x_grid, kde(x_grid), linewidth=1.5, alpha=0.7,
                             label=f"{variant_key} (Avg: {avg_wait_str} min, Max: {max_wait_str} min)")
                except Exception as e:
                    print(f"Could not plot KDE for {variant_key}: {e}")
                    plt.plot([], [], linewidth=1.5, alpha=0.7,
                             label=f"{variant_key} (Avg: {avg_wait_str} min, Max: {max_wait_str} min, KDE Error)")

            elif len(waits_min) == 1:  # Handle single point case
                avg_wait_str = f"{waits_min[0]:.2f}"
                max_wait_str = f"{waits_min[0]:.2f}"
                plt.plot([], [],
                         label=f"{variant_key} (Avg: {avg_wait_str} min, Max: {max_wait_str} min, Single Point)")
            else:  # No data
                plt.plot([], [], label=f"{variant_key} (Avg: N/A, Max: N/A, No Data)")

        plt.xlabel('Waiting Time (minutes)')
        plt.ylabel('Density')
        title_text = f'Smoothed Waiting Time Distribution: {base_key} and Variants'
        if not sorted_variants and base_key in results:
            title_text = f'Smoothed Waiting Time Distribution: {base_key}'

        plt.title(title_text)
        plt.legend(fontsize='small')
        plt.grid(True)
        plt.xlim(0, max_minutes)
        plt.ylim(bottom=0)  # Ensure density is not negative
        plt.show()


if __name__ == "__main__":
    excel_file = './data/input V1.xlsx'
    demand, _ = load_excel_data(excel_file)
    plot_demand(demand)

    _, results = run_simulation(excel_file)
    plot_waiting_times(results)