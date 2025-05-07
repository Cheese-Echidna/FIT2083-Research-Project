import pandas as pd
import numpy as np
import simpy

class Station:
    def __init__(self, env):
        self.env = env
        self.queue = []             # store arrival times
        self.waiting_times = []     # store computed waiting times

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
    Sheet1 (index 0): 'Time' (excel time, e.g., datetime.time), 'Sunday', 'Monday-Thursday Average'
    Sheet2 (index 1): 'Sunday Inbound', 'Monday-Thursday Inbound'

    Returns:
      demand: dict of DataFrames with columns ['hour', 'busyness'] for each scenario key
      timetables: dict of lists of departure times (seconds since midnight) per scenario
    """
    # Read both sheets
    df_busyness = pd.read_excel(path, sheet_name=0)
    df_tt = pd.read_excel(path, sheet_name=1)

    # Parse and normalize Time column for busyness
    # Handles datetime.time, string with AM/PM, or numeric hours
    times_raw = df_busyness['Time']
    # Convert all to datetime64 (date component ignored)
    times_dt = pd.to_datetime(times_raw.astype(str), errors='coerce')
    times_sec = times_dt.dt.hour * 3600 + times_dt.dt.minute * 60 + times_dt.dt.second

    # Build demand dict for each scenario
    demand = {}
    mapping = [('Sunday', 'Sunday'), ('Monday-Thursday average', 'MonThu')]
    for col, key in mapping:
        df = pd.DataFrame({
            'hour': (times_sec // 3600).astype(int),
            'busyness': df_busyness[col].astype(float)
        })
        demand[key] = df

    # Build timetables dict for each scenario
    timetables = {}
    mapping_tt = [('Sunday inbound', 'Sunday'), ('Monday-Thursday inbound', 'MonThu')]
    for col, key in mapping_tt:
        times_raw = df_tt[col].dropna()
        # Convert each entry to datetime, handling AM/PM
        secs = []
        for t in times_raw:
            dt = pd.to_datetime(str(t), errors='coerce')
            if pd.isna(dt):
                # try common AM/PM format
                dt = pd.to_datetime(str(t), format='%I:%M:%S %p', errors='coerce')
            secs.append(dt.hour * 3600 + dt.minute * 60 + dt.second)
        timetables[key] = sorted(secs)

    return demand, timetables

def passenger_generator(env: simpy.Environment, station: Station,
                        demand_df: pd.DataFrame, total_passengers: int):
    """
    Generate passenger arrivals over a 24h period based on demand weights.
    - demand_df: DataFrame with 'hour' and 'busyness'
    - total_passengers: absolute count to scale percentages to numbers
    """
    # Normalize hourly weights
    weights = demand_df['busyness'] / demand_df['busyness'].sum()
    for idx, row in demand_df.iterrows():
        hour = row['hour']
        n_arrivals = int(np.round(weights[idx] * total_passengers))
        # arrival times uniformly within the hour window
        start = hour * 3600
        end = (hour + 1) * 3600
        arrival_times = np.random.uniform(start, end, size=n_arrivals)
        for at in np.sort(arrival_times):
            yield env.timeout(at - env.now)
            station.arrive(at)


def train_process(env: simpy.Environment, station: Station,
                  timetable_sec: list[float]):
    """
    Simulate trains departing according to timetable_sec (seconds since midnight).
    """
    for departure in timetable_sec:
        yield env.timeout(departure - env.now)
        station.board_train(departure)


def run_simulation(excel_path: str, passengers_mult: int):
    """
    Run simulations for Sunday and Monday-Thursday scenarios using the provided Excel.

    Returns:
      dict of waiting time lists by scenario key ('Sunday', 'MonThu')
    """
    demand, timetables = load_excel_data(excel_path)
    results = {}
    for scenario in ['Sunday', 'MonThu']:
        total_percentage = demand[scenario]['busyness'].sum()
        total_passengers = int(passengers_mult * total_percentage)

        env = simpy.Environment()
        station = Station(env)

        # start processes
        env.process(passenger_generator(env, station,
                                        demand[scenario], total_passengers))
        env.process(train_process(env, station,
                                  timetables[scenario]))

        # run full day
        env.run(until=24 * 3600)
        results[scenario] = station.waiting_times

    return results

if __name__ == "__main__":
    # Example
    excel_file = '../Data/input V1.xlsx'
    total_per_day = 5000  # set your daily inbound count
    waiting = run_simulation(excel_file, total_per_day)
    # waiting['Sunday'], waiting['MonThu'] contain lists of wait times (s)