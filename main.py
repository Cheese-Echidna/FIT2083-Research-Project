import pathlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
import simpy
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# NOTE: Subtracted 6% baseline from all scores

def time_to_seconds(t: pd.Series) -> pd.Series:
    """Convert datetime-like or string series to seconds since midnight."""
    times = pd.to_datetime(t.astype(str), errors='coerce')
    return times.dt.hour * 3600 + times.dt.minute * 60 + times.dt.second


@dataclass
class Station:
    env: simpy.Environment
    queue: List[float] = field(default_factory=list)
    waiting_times: List[float] = field(default_factory=list)

    def arrive(self, arrival_time: float) -> None:
        """Record a passenger arrival at the current simulation time."""
        self.queue.append(self.env.now)

    def board(self, departure_time: float) -> None:
        """Board all passengers whose arrival <= departure and record wait times."""
        onboard, remaining = [], []
        for t in self.queue:
            if t <= departure_time:
                self.waiting_times.append(departure_time - t)
                onboard.append(t)
            else:
                remaining.append(t)
        self.queue = remaining


def load_excel_data(
        path: pathlib.Path,
        busyness_sheet: int = 0,
        timetable_sheet: int = 1
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[float]]]:
    df_bus = pd.read_excel(path, sheet_name=busyness_sheet)
    df_tt = pd.read_excel(path, sheet_name=timetable_sheet)

    times_sec = time_to_seconds(df_bus['Time'])
    demand = {
        key: pd.DataFrame({
            'hour': (times_sec // 3600).astype(int),
            'busyness': df_bus[col].astype(float)
        })
        for col, key in [('Sunday', 'Sunday'), ('Monday-Thursday Average', 'MonThu')]
    }

    timetables = {}
    for col, key in [('Sunday Inbound', 'Sunday'), ('Monday-Thursday Inbound', 'MonThu')]:
        secs = (
            df_tt[col]
            .dropna()
            .apply(lambda x: time_to_seconds(pd.Series([x]))[0])
        )
        timetables[key] = sorted(secs.tolist())

    return demand, timetables


def redistribute_even(count: int, start: float, end: float) -> List[float]:
    if count <= 0:
        return []
    interval = (end - start) / count
    return [start + i * interval for i in range(count)]


def generate_even_timetable(
        original: List[float],
        start: float = 0,
        end: float = 24 * 3600
) -> List[float]:
    return redistribute_even(len(original), start, end)


def generate_weighted_timetable(
        demand: pd.DataFrame,
        count: int,
        start_hour: int = 0,
        end_hour: int = 24
) -> List[float]:
    # 1) select only the hours we care about
    period = demand[
        (demand['hour'] >= start_hour) &
        (demand['hour'] < end_hour)
        ].copy()
    if period.empty or count <= 0:
        return []

    # 2) compute “raw” (float) allocations
    total_busyness = period['busyness'].sum()
    period['raw'] = period['busyness'] / total_busyness * count

    # 3) floor to get initial integer allocs
    period['alloc'] = np.floor(period['raw']).astype(int)

    # 4) enforce at least 1 train if busyness>0
    mask = (period['busyness'] > 0) & (period['alloc'] < 1)
    period.loc[mask, 'alloc'] = 1

    # 5) adjust so sum(alloc) == count
    allocated = period['alloc'].sum()

    if allocated > count:
        # too many: remove from the lowest-busyness hours first
        to_remove = int(allocated - count)
        drop_order = period[period['alloc'] > 1] \
            .sort_values('busyness') \
            .head(to_remove) \
            .index
        period.loc[drop_order, 'alloc'] -= 1

    elif allocated < count:
        # too few: distribute remainder by highest fractional‐part, then busyness
        to_add = int(count - allocated)
        period['frac'] = period['raw'] - period['alloc']
        add_order = period.sort_values(
            ['frac', 'busyness'],
            ascending=[False, False]
        ).head(to_add).index
        period.loc[add_order, 'alloc'] += 1

    # 6) now build the departure‐times list by hour
    times: List[float] = []
    for _, row in period.iterrows():
        h = int(row['hour'])
        k = int(row['alloc'])
        if k == 0:
            continue

        hour_start = h * 3600
        hour_end   = (h + 1) * 3600

        if k == 1:
            # singleton in the middle of the hour
            times.append(hour_start + 0.5 * (hour_end - hour_start))
        else:
            # even spacing from start→end inclusive
            interval = (hour_end - hour_start) / (k - 1)
            times.extend(hour_start + i * interval for i in range(k))

    return sorted(times)




def generate_minmax_timetable(original: List[float]) -> List[float]:
    if not original:
        return []
    return redistribute_even(len(original), original[0], original[-1])


def passenger_generator(env, station, demand, total):
    weights = demand['busyness'] / demand['busyness'].sum()
    # initialize counters
    generated_per_hour = {hour: 0 for hour in demand['hour']}
    for hour, weight in zip(demand['hour'], weights):
        n = int(np.floor(weight * total))
        generated_per_hour[hour] += n
        # we will distribute the remainder below
        times = np.sort(np.random.rand(n) * 3600 + hour * 3600)
        for t in times:
            yield env.timeout(max(0, t - env.now))
            station.arrive(t)
    # distribute leftover passengers (so sum == total)
    remainder = total - sum(generated_per_hour.values())
    # pick top‐weight hours to allocate the leftover one by one
    top_hours = demand.sort_values('busyness', ascending=False)['hour'].tolist()
    for i in range(remainder):
        h = top_hours[i % len(top_hours)]
        generated_per_hour[h] += 1
        t = h * 3600 + np.random.rand() * 3600
        yield env.timeout(max(0, t - env.now))
        station.arrive(t)

    # sanity check
    assert sum(generated_per_hour.values()) == total, \
        f"Generated {sum(generated_per_hour.values())} ≠ expected {total}"



def train_process(env: simpy.Environment, station: Station, times: List[float]) -> None:
    """Dispatch trains at scheduled times, with safe timeouts."""
    for t in times:
        delay = t - env.now
        yield env.timeout(max(delay, 0))
        station.board(t)


def run_simulation(
        excel_path: str,
        total_passengers_factor: float = 10_000.0
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Run simulation for real and generated timetables.
    Returns demand dict, all timetables, and waiting times per scenario.
    """
    np.random.seed(42)
    path = pathlib.Path(excel_path)
    demand, timetables = load_excel_data(path)

    # create variants
    generated: Dict[str, List[float]] = {}
    for key, original in timetables.items():
        generated[f"{key}_even24"] = generate_even_timetable(original)
        # generated[f"{key}_even6_24"] = generate_even_timetable(original, 6 * 3600)
        generated[f"{key}_weighted"] = generate_weighted_timetable(
            demand[key], len(original)
        )
        generated[f"{key}_even_over_same_time"] = generate_minmax_timetable(original)

    all_tt = {**timetables, **generated}
    results: Dict[str, List[float]] = {}

    for scenario, times in all_tt.items():
        base = scenario.split('_')[0]
        total = int(total_passengers_factor * demand[base]['busyness'].sum())

        env = simpy.Environment()
        station = Station(env)
        env.process(passenger_generator(env, station, demand[base], total))
        env.process(train_process(env, station, times))
        env.run(until=24 * 3600)

        # assume leftover passengers board on first train next morning
        if station.queue and times:
            next_dep = times[0] + 24 * 3600
            for arrival in station.queue:
                station.waiting_times.append(next_dep - arrival)
            station.queue.clear()

        results[scenario] = station.waiting_times

    return demand, all_tt, results


def plot_demand(demand: Dict[str, pd.DataFrame]) -> None:
    """Plot hourly busyness for each scenario."""
    plt.figure()
    for key, df in demand.items():
        plt.plot(df['hour'], df['busyness'], marker='o', label=key)
    plt.xlabel('Hour of Day')
    plt.ylabel('Busyness')
    plt.title('Hourly Busyness')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_waiting_times(
        timetables: Dict[str, List[float]],
        results: Dict[str, List[float]]
) -> None:
    """Plot KDE of waiting times and train schedules with hourly & 6-hour grid."""
    MAX_MIN = 60
    x = np.linspace(0, MAX_MIN, 500)

    grouped: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    for key, waits in results.items():
        base = key.split('_')[0]
        grouped.setdefault(base, []).append((key, np.array(waits) / 60.0))

    for base, series in grouped.items():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        colors = plt.cm.tab10(np.linspace(0, 1, len(series)))

        # ---- plot the KDEs as before ----
        for (name, data), color in zip(series, colors):
            if len(data) >= 2:
                kde = gaussian_kde(data)
                label = (
                    f"{name} (avg={data.mean():.1f}, "
                    f"p90={np.percentile(data, 90):.0f}, "
                    f"p95={np.percentile(data, 95):.0f}, "
                    f"p99={np.percentile(data, 99):.0f})"
                )
                ax1.plot(x, kde(x), label=label, color=color)
        ax1.set_xlabel('Waiting Time (min)')
        ax1.set_ylabel('Density')
        ax1.set_title(f'KDE of Waiting Time Distributions: {base}')
        ax1.set_xlim(0, MAX_MIN)
        ax1.grid(True)
        ax1.legend(fontsize='small')

        # Timeline plot
        for idx, (name, data) in enumerate(series):
            train_times = [t / 3600 for t in sorted(timetables[name])]
            y_pos = idx + 1
            ax2.scatter(train_times, [y_pos] * len(train_times),
                        color=colors[idx], label=name)

        # hourly & 6-hour grid lines
        ax2.set_xlim(0, 24)
        ax2.set_ylim(0.5, len(series) + 0.5)
        # minor ticks every 1 hour, major every 6 hours
        ax2.set_xticks(np.arange(0, 25, 1), minor=True)
        ax2.set_xticks(np.arange(0, 25, 6), minor=False)
        # draw grids
        ax2.grid(which='minor', color='lightgrey', linestyle='-', linewidth=0.5)
        ax2.grid(which='major', color='grey', linestyle='-', linewidth=1.0)

        ax2.set_yticks(range(1, len(series) + 1))
        ax2.set_yticklabels([name for name, _ in series])
        ax2.set_xlabel('Time (hour)')
        ax2.set_ylabel('Timetable Variant')
        ax2.set_title(f'Train Schedule Timeline: {base}')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    FILE = './data/input V1.xlsx'
    demand_data, all_tt, sim_results = run_simulation(FILE)
    plot_demand(demand_data)
    plot_waiting_times(all_tt, sim_results)
