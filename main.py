import pandas as pd
import numpy as np
import simpy
import matplotlib.pyplot as plt


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
    Sheet1 (index 0): 'Time' (excel time, e.g., datetime.time or string like '3:45:00 PM'), 'Sunday', 'Monday-Thursday Average'
    Sheet2 (index 1): 'Sunday Inbound', 'Monday-Thursday Inbound'

    Returns:
      demand: dict of DataFrames with columns ['hour', 'busyness'] for each scenario key
      timetables: dict of lists of departure times (seconds since midnight) per scenario
    """
    df_busyness = pd.read_excel(path, sheet_name=0)
    df_tt = pd.read_excel(path, sheet_name=1)

    # Parse and normalize Time column for busyness
    times_raw = df_busyness['Time']
    # Replace the problematic line:
    # times_dt = pd.to_datetime(times_raw.dt.strftime('%I:%M:%S %p'), format='%I:%M:%S %p')
    # With this corrected line:
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
        times_raw = df_tt[col].dropna()
        secs = []
        for t in times_raw:
            dt = pd.to_datetime(str(t), errors='coerce')
            if pd.isna(dt):
                dt = pd.to_datetime(str(t), format='%I:%M:%S %p', errors='coerce')
            secs.append(dt.hour * 3600 + dt.minute * 60 + dt.second)
        timetables[key] = sorted(secs)

    return demand, timetables


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
    Run simulations for Sunday and Monday-Thursday scenarios.
    Total passengers per day = 10,000 * sum of busyness weights.

    Returns:
      dict of waiting time lists by scenario ('Sunday', 'MonThu')
    """
    demand, timetables = load_excel_data(excel_path)
    results = {}
    for scenario in ['Sunday', 'MonThu']:
        total_percentage = demand[scenario]['busyness'].sum()
        total_passengers = int(10000 * total_percentage)
        env = simpy.Environment()
        station = Station(env)
        env.process(passenger_generator(env, station,
                                        demand[scenario], total_passengers))
        env.process(train_process(env, station,
                                  timetables[scenario]))
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
    """Plot waiting time distributions for each scenario (in minutes)."""
    plt.figure()
    # Set a reasonable maximum limit for x-axis (e.g., 60 minutes)
    max_minutes = 60
    bins = np.linspace(0, max_minutes, 120)  # 120 bins for good resolution

    # Define colors for each scenario
    colors = {'Sunday': 'blue', 'MonThu': 'orange'}

    for key, waits in results.items():
        waits_min = np.array(waits) / 60
        avg_wait = np.mean(waits_min)

        # Plot histogram
        plt.hist(waits_min, bins=bins, alpha=0.5, label=f"{key}", color=colors.get(key, None))

        # Add vertical line for average
        plt.axvline(x=avg_wait, color=colors.get(key, None), linestyle='--',
                    linewidth=2, label=f"{key} Avg: {avg_wait:.2f} min")

    # Calculate and plot global average
    all_waits = []
    for waits in results.values():
        all_waits.extend([w/60 for w in waits])
    global_avg = np.mean(all_waits)
    plt.axvline(x=global_avg, color='black', linestyle='-', 
                linewidth=3, label=f"Global Avg: {global_avg:.2f} min")
    
    plt.xlabel('Waiting Time (minutes)')
    plt.ylabel('Frequency')
    plt.title('Waiting Time Distribution by Scenario')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    excel_file = './data/input V1.xlsx'
    # Load data and timetables
    demand, _ = load_excel_data(excel_file)
    plot_demand(demand)

    # Run simulation and get results
    _, results = run_simulation(excel_file)
    plot_waiting_times(results)
