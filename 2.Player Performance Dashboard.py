import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
def bat_an(name):
    path = r"C:\Users\sakha\Downloads\PythonProject\Codes\cleaned_ball_by_ball.csv"
    try:
        df = pd.read_csv(path, low_memory=False)
    except:
        return None, None, None
    name = name.upper()
    bat_df = df[df['batter'].str.replace(" ", "") == name].copy()
    if bat_df.empty:
        return None, None, None
    bat_df['over_num'] = bat_df['overs'] + 1
    valid = bat_df[bat_df['extra_type'] != 'wides']
    ovr = valid.groupby('over_num').agg({
        'batsman_run': 'sum',
        'ballnumber': 'count',
        'isWicketDelivery': lambda x: ((x == 1) & (bat_df.loc[x.index, 'player_out'].str.replace(" ", "") == name)).sum()
    }).reindex(range(1, 21), fill_value=0).reset_index()
    ovr.columns = ['over_num', 'runs', 'balls', 'dismissals']
    ovr['avg'] = (ovr['runs'] / ovr['dismissals'].replace(0, np.nan)).fillna(0).round(2)
    ovr['sr'] = (ovr['runs'] / ovr['balls'].replace(0, np.nan) * 100).fillna(0).round(2)
    ovr['runs'] = ovr['runs'].astype(int)
    ovr['balls'] = ovr['balls'].astype(int)
    ovr['dismissals'] = ovr['dismissals'].astype(int)
    ovr = ovr.reset_index().rename(columns={'index': 'over_idx'})
    ph = pd.DataFrame({
        'Phase': ['(1-6)', '(7-15)', '(16-20)'],
        'Overs': ['1-6', '7-15', '16-20'],
        'Runs': [
            ovr[ovr['over_num'].between(1, 6)]['runs'].sum(),
            ovr[ovr['over_num'].between(7, 15)]['runs'].sum(),
            ovr[ovr['over_num'].between(16, 20)]['runs'].sum()
        ],
        'Balls': [
            ovr[ovr['over_num'].between(1, 6)]['balls'].sum(),
            ovr[ovr['over_num'].between(7, 15)]['balls'].sum(),
            ovr[ovr['over_num'].between(16, 20)]['balls'].sum()
        ],
        'Dismissals': [
            ovr[ovr['over_num'].between(1, 6)]['dismissals'].sum(),
            ovr[ovr['over_num'].between(7, 15)]['dismissals'].sum(),
            ovr[ovr['over_num'].between(16, 20)]['dismissals'].sum()
        ]
    })
    total = ph['Runs'].sum()
    ph['Percentage'] = (ph['Runs'] / total * 100).round(2) if total > 0 else 0
    ph['Strike_Rate'] = (ph['Runs'] / ph['Balls'].replace(0, np.nan) * 100).fillna(0).round(2)
    ph['Average'] = (ph['Runs'] / ph['Dismissals'].replace(0, np.nan)).fillna(0).round(2)
    ph['Runs'] = ph['Runs'].astype(int)
    ph['Balls'] = ph['Balls'].astype(int)
    ph['Dismissals'] = ph['Dismissals'].astype(int)
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(111)
    bars = ax1.bar(ovr['over_num'], ovr['runs'], color='orange', alpha=0.7, label='Runs')
    ax1.set_xlabel('Over Number', fontsize=12)
    ax1.set_ylabel('Runs Scored', fontsize=12, color='orange')
    ax1.tick_params(axis='y', labelcolor='orange')
    ax1.set_title(f'{name}: Runs, Average, and Strike Rate per Over', fontsize=14)
    ax1.set_xticks(range(1, 21))
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    for bar, runs in zip(bars, ovr['runs']):
        if runs > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{runs}',
                     ha='center', va='bottom', fontsize=10)
    for bar, avg, runs in zip(bars, ovr['avg'], ovr['runs']):
        if runs > 0:
            text = f'Avg: {avg:.1f}'
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, text,
                     ha='center', va='center', fontsize=8, color='black', rotation=90,
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    ax2 = ax1.twinx()
    ax2.plot(ovr['over_num'], ovr['sr'], color='red', marker='o', label='Strike Rate')
    ax2.set_ylabel('Strike Rate', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    ph_data = ph[['Phase', 'Runs', 'Balls', 'Dismissals', 'Percentage', 'Strike_Rate', 'Average']].round(2)
    ax3 = fig.add_subplot(111, frame_on=False)
    ax3.xaxis.set_visible(False)
    ax3.yaxis.set_visible(False)
    table = ax3.table(cellText=ph_data.values, colLabels=ph_data.columns, loc='bottom', bbox=[0, -0.5, 1, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(20)
    for key, cell in table.get_celld().items():
        cell.set_text_props(fontweight='bold')
    fig.subplots_adjust(bottom=0.35)
    plt.tight_layout()
    try:
        ovr.to_csv('batter_over_stats.csv', index=False)
    except:
        pass
    return fig, ovr, ph
def proc():
    try:
        df = pd.read_csv(r'C:\Users\sakha\Downloads\PythonProject\Codes\cleaned_ball_by_ball.csv')
        mdf = pd.read_csv(r'C:\Users\sakha\Downloads\PythonProject\Codes\cleaned_match_level.csv')
    except:
        return None
    if 'Season' not in mdf.columns:
        if 'Date' in mdf.columns:
            try:
                mdf['Season'] = pd.to_datetime(mdf['Date']).dt.year
            except:
                return None
        else:
            return None
    try:
        mdf['Season'] = mdf['Season'].astype(str).str.strip().astype(int)
    except:
        return None
    df = df.merge(mdf[['ID', 'Season']], on='ID', how='left')
    if 'Season' not in df.columns or df['Season'].isna().all():
        return None
    df['Season'] = df['Season'].fillna(0).astype(int)
    def to_overs(balls):
        return balls // 6
    bat_df = df[df['extra_type'] != 'wides']
    bat = bat_df.groupby(['batter', 'Season']).agg({
        'batsman_run': lambda x: int(x.sum()),
        'ballnumber': lambda x: int(x.count()),
        'isWicketDelivery': lambda x: int((x & (bat_df.loc[x.index, 'player_out'] == bat_df.loc[x.index, 'batter'])).sum()),
        'ID': lambda x: int(x.nunique())
    }).rename(columns={'batsman_run': 'runs', 'ballnumber': 'balls', 'isWicketDelivery': 'dismissals', 'ID': 'matches'}).reset_index()
    fours = bat_df[bat_df['batsman_run'] == 4].groupby(['batter', 'Season'])['batsman_run'].count().reindex(
        bat.set_index(['batter', 'Season']).index, fill_value=0).reset_index(name='fours')
    sixes = bat_df[bat_df['batsman_run'] == 6].groupby(['batter', 'Season'])['batsman_run'].count().reindex(
        bat.set_index(['batter', 'Season']).index, fill_value=0).reset_index(name='sixes')
    bat = bat.merge(fours[['batter', 'Season', 'fours']], on=['batter', 'Season'], how='left').merge(
        sixes[['batter', 'Season', 'sixes']], on=['batter', 'Season'], how='left').fillna({'fours': 0, 'sixes': 0})
    bat['fours'] = bat['fours'].astype(int)
    bat['sixes'] = bat['sixes'].astype(int)
    bat['sr'] = (bat['runs'] / bat['balls'] * 100).round(2)
    bat['avg'] = (bat['runs'] / bat['dismissals'].replace(0, np.nan)).round(2)
    bat.fillna({'sr': 0, 'avg': 0}, inplace=True)
    bat = bat.reset_index().rename(columns={'index': 'bat_idx'})
    bowl_df = df[df['extra_type'] != 'wides']
    wkt_types = ['caught', 'bowled', 'lbw', 'stumped', 'hit wicket']
    wkt_df = df[df['kind'].isin(wkt_types)]
    wkts = wkt_df.groupby(['bowler', 'Season'])['isWicketDelivery'].sum().reset_index(name='wickets')
    wkts['wickets'] = wkts['wickets'].astype(int)
    bowl = bowl_df.groupby(['bowler', 'Season']).agg({
        'ballnumber': lambda x: int(x.count()),
        'ID': lambda x: int(x.nunique())
    }).rename(columns={'ballnumber': 'balls_bowled', 'ID': 'matches'}).reset_index()
    bowl = bowl.merge(wkts, on=['bowler', 'Season'], how='left').fillna({'wickets': 0})
    bowl['wickets'] = bowl['wickets'].astype(int)
    runs_conceded = df.groupby(['bowler', 'Season'])['total_run'].sum().reset_index(name='runs_conceded')
    runs_conceded['runs_conceded'] = runs_conceded['runs_conceded'].astype(int)
    bowl = bowl.merge(runs_conceded, on=['bowler', 'Season'], how='left').fillna({'runs_conceded': 0})
    bowl['runs_conceded'] = bowl['runs_conceded'].astype(int)
    bowl['overs'] = bowl['balls_bowled'].apply(to_overs).astype(int)
    bowl['econ'] = (bowl['runs_conceded'] / bowl['overs'].replace(0, np.nan)).round(2)
    bowl['bowl_avg'] = (bowl['runs_conceded'] / bowl['wickets'].replace(0, np.nan)).round(2)
    bowl['bowl_sr'] = (bowl['balls_bowled'] / bowl['wickets'].replace(0, np.nan)).round(2)
    bowl.fillna({'econ': 0, 'bowl_avg': 0, 'bowl_sr': 0}, inplace=True)
    bowl = bowl.reset_index().rename(columns={'index': 'bowl_idx'})
    field = df[df['kind'] == 'caught'].groupby(['fielders_involved', 'Season'])['ID'].count().reset_index().rename(
        columns={'fielders_involved': 'player', 'ID': 'catches'})
    field['catches'] = field['catches'].astype(int)
    field = field.reset_index().rename(columns={'index': 'field_idx'})
    pom = mdf.groupby(['Player_of_Match', 'Season'])['ID'].count().reset_index().rename(
        columns={'Player_of_Match': 'player', 'ID': 'pom'})
    pom['pom'] = pom['pom'].astype(int)
    pom = pom.reset_index().rename(columns={'index': 'pom_idx'})
    all_p = pd.concat([
        bat[['batter', 'Season']].rename(columns={'batter': 'player'}),
        bowl[['bowler', 'Season']].rename(columns={'bowler': 'player'}),
        field[['player', 'Season']],
        pom[['player', 'Season']]
    ]).drop_duplicates().reset_index(drop=True)
    stats = all_p.merge(bat.rename(columns={'batter': 'player'}), on=['player', 'Season'], how='left').merge(
        bowl.rename(columns={'bowler': 'player'}), on=['player', 'Season'], how='left').merge(
        field, on=['player', 'Season'], how='left').merge(pom, on=['player', 'Season'], how='left').fillna(0)
    for col in ['runs', 'balls', 'dismissals', 'fours', 'sixes', 'wickets', 'balls_bowled', 'runs_conceded', 'overs',
                'catches', 'pom', 'bat_idx', 'bowl_idx', 'field_idx', 'pom_idx']:
        if col in stats.columns:
            stats[col] = stats[col].astype(int)
    stats['p_clean'] = stats['player'].str.replace(" ", "")
    stats = stats.reset_index().rename(columns={'index': 'p_idx'})
    stats.to_csv('player_performance_stats.csv', index=False)
    return stats
def sel_p(p_list):
    p_list = sorted(p_list)
    pmap = {p: p.replace(" ", "") for p in p_list}
    prefix = ""
    print("\nAvailable Players (type letters to filter, spaces are ignored, 'exit' to quit):")
    while True:
        fltr = [p for p in p_list if pmap[p].startswith(prefix)]
        if not fltr:
            print("No players match the current input.")
            prefix = prefix[:-1] if prefix else ""
            continue
        if len(fltr) == 1:
            print(f"Automatically selecting: {fltr[0]}")
            return pmap[fltr[0]]
        print("\nMatching Players:")
        for i, p in enumerate(fltr[:10]):
            print(f"{i}: {p}")
        if len(fltr) > 10:
            print(f"... and {len(fltr) - 10} more")
        print(f"\nCurrent input: {prefix}")
        print("Enter letters to filter, number to select, 'back' to remove last letter, or 'exit':")
        inp = input().strip()
        if inp.lower() == 'exit':
            print("Exiting program.")
            sys.exit(0)
        inp = inp.replace(" ", "").upper()
        if inp == 'BACK':
            prefix = prefix[:-1] if prefix else ""
        elif inp.isdigit():
            idx = int(inp)
            if 0 <= idx < len(fltr):
                return pmap[fltr[idx]]
            else:
                print("Invalid number. Try again.")
        elif inp:
            prefix += inp
        else:
            print("Invalid input. Enter letters, number, 'back', or 'exit'.")
def sel_g():
    print("\nSelect graphs to display (enter numbers together, e.g., '123' for graphs 1, 2, 3, 'exit' to return to player selection):")
    print("1: Batting")
    print("2: Bowling")
    print("3: Fielding")
    print("4: P.O.M")
    print("5: Over-by-Over")
    while True:
        inp = input("Enter graph numbers: ").strip()
        if inp.lower() == 'exit':
            return None
        if not inp.isdigit():
            print("Invalid input. Enter numbers like '123' or 'exit'.")
            continue
        ch = sorted(set(int(x) for x in inp if x.isdigit()))
        if not ch:
            print("No numbers entered. Try again or type 'exit'.")
            continue
        if all(1 <= x <= 5 for x in ch):
            return ch
        else:
            print("Invalid numbers. Use 1 to 5 only.")
def plot_bat(d, name, sr):
    fd = pd.DataFrame({'Season': sr})
    if not d.empty:
        d = d.set_index('Season').reindex(sr, fill_value=0).reset_index()
    else:
        d = fd.copy()
        for col in ['runs', 'matches', 'balls', 'sr', 'avg', 'fours', 'sixes']:
            d[col] = 0
        d['sr'] = d['sr'].astype(float)
        d['avg'] = d['avg'].astype(float)
    fig = plt.figure(figsize=(6, 4))
    ax1 = fig.add_subplot(111)
    ax1.bar(d['Season'], d['runs'], color='royalblue', label='Runs', alpha=0.7)
    ax1.set_xlabel('Season', fontsize=12)
    ax1.set_ylabel('Runs', fontsize=12, color='royalblue')
    ax1.tick_params(axis='y', labelcolor='royalblue')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_xticks(sr)
    ax1.set_xticklabels(sr, rotation=45)
    ax2 = ax1.twinx()
    ax2.plot(d['Season'], d['sr'], color='orange', marker='o', label='Strike Rate')
    ax2.set_ylabel('Strike Rate', fontsize=12, color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    fig.suptitle(f'{name}: Batting - Runs and Strike Rate', fontsize=14)
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    plt.tight_layout()
    return fig, d
def plot_bowl(d, name, sr):
    fd = pd.DataFrame({'Season': sr})
    if not d.empty:
        d = d.set_index('Season').reindex(sr, fill_value=0).reset_index()
    else:
        d = fd.copy()
        for col in ['wickets', 'matches', 'overs', 'runs_conceded']:
            d[col] = 0
        for col in ['econ', 'bowl_avg', 'bowl_sr']:
            d[col] = 0.0
    fig = plt.figure(figsize=(6, 4))
    ax1 = fig.add_subplot(111)
    ax1.bar(d['Season'], d['wickets'], color='green', label='Wickets', alpha=0.7)
    ax1.set_xlabel('Season', fontsize=12)
    ax1.set_ylabel('Wickets', fontsize=12, color='green')
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_xticks(sr)
    ax1.set_xticklabels(sr, rotation=45)
    ax2 = ax1.twinx()
    ax2.plot(d['Season'], d['econ'], color='red', marker='o', label='Economy')
    ax2.set_ylabel('Economy', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    fig.suptitle(f'{name}: Bowling - Wickets and Economy', fontsize=14)
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    plt.tight_layout()
    return fig, d
def plot_field(d, name, sr):
    fd = pd.DataFrame({'Season': sr})
    if not d.empty:
        d = d.set_index('Season').reindex(sr, fill_value=0).reset_index()
    else:
        d = fd.copy()
        d['catches'] = 0
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.bar(d['Season'], d['catches'], color='purple', alpha=0.7)
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Catches', fontsize=12)
    ax.set_title(f'{name}: Fielding - Catches', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xticks(sr)
    ax.set_xticklabels(sr, rotation=45)
    plt.tight_layout()
    return fig, d
def plot_pom(d, name, sr):
    fd = pd.DataFrame({'Season': sr})
    if not d.empty:
        d = d.set_index('Season').reindex(sr, fill_value=0).reset_index()
    else:
        d = fd.copy()
        d['pom'] = 0
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.bar(d['Season'], d['pom'], color='gold', alpha=0.7)
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('P.O.M Awards', fontsize=12)
    ax.set_title(f'{name}: P.O.M Awards', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xticks(sr)
    ax.set_xticklabels(sr, rotation=45)
    plt.tight_layout()
    return fig, d
def main():
    stats = proc()
    if stats is None:
        print("Failed to load datasets.")
        return
    p_list = stats['player'].unique()
    while True:
        name = sel_p(p_list)
        if not name:
            print("No player selected.")
            continue
        if name not in stats['p_clean'].unique():
            print(f"Player '{name}' not found in dataset.")
            continue
        ch = sel_g()
        if ch is None:
            print("Returning to player selection.")
            continue
        if not ch:
            print("No graphs selected.")
            continue
        p_data = stats[stats['p_clean'] == name]
        seasons = p_data['Season'].unique()
        first = min(seasons)
        last = max(seasons)
        sr = list(range(first, last + 1))
        matches = int(p_data['matches_x'].sum() + p_data['matches_y'].sum())
        runs = int(p_data['runs'].sum())
        wickets = int(p_data['wickets'].sum())
        catches = int(p_data['catches'].sum())
        pom = int(p_data['pom'].sum())
        orig_name = p_data['player'].iloc[0]
        print(f"\nPlayer Summary for {orig_name}:")
        print(f"Matches: {matches}")
        print(f"Runs: {runs}")
        print(f"Wickets: {wickets}")
        print(f"Catches: {catches}")
        print(f"P.O.M Awards: {pom}")
        figs = {}
        bat_d = p_data[p_data['runs'] > 0][['Season', 'matches_x', 'runs', 'balls', 'sr', 'avg', 'fours', 'sixes']].rename(
            columns={'matches_x': 'matches'})
        bat_d = bat_d.sort_values('Season')
        if 1 in ch:
            figs[1], bat_full = plot_bat(bat_d, orig_name, sr)
        else:
            fd = pd.DataFrame({'Season': sr})
            bat_full = bat_d.set_index('Season').reindex(sr, fill_value=0).reset_index() if not bat_d.empty else fd.assign(
                runs=0, matches=0, balls=0, sr=0.0, avg=0.0, fours=0, sixes=0)
        if not bat_full.empty:
            print("\nBatting Statistics:")
            print(bat_full[['Season', 'matches', 'runs', 'balls', 'sr', 'avg', 'fours', 'sixes']].to_string(index=False))
        else:
            print("\nNo batting stats available.")
        bowl_d = p_data[p_data['wickets'] > 0][
            ['Season', 'matches_y', 'wickets', 'overs', 'runs_conceded', 'econ', 'bowl_avg', 'bowl_sr']].rename(
            columns={'matches_y': 'matches'})
        bowl_d = bowl_d.sort_values('Season')
        if 2 in ch:
            figs[2], bowl_full = plot_bowl(bowl_d, orig_name, sr)
        else:
            fd = pd.DataFrame({'Season': sr})
            bowl_full = bowl_d.set_index('Season').reindex(sr, fill_value=0).reset_index() if not bowl_d.empty else fd.assign(
                wickets=0, matches=0, overs=0, runs_conceded=0, econ=0.0, bowl_avg=0.0, bowl_sr=0.0)
        if not bowl_full.empty:
            print("\nBowling Statistics:")
            print(bowl_full[['Season', 'matches', 'wickets', 'overs', 'runs_conceded', 'econ', 'bowl_avg', 'bowl_sr']].to_string(
                index=False))
        else:
            print("\nNo bowling stats available.")
        field_d = p_data[p_data['catches'] > 0][['Season', 'catches']]
        field_d = field_d.sort_values('Season')
        if 3 in ch:
            figs[3], field_full = plot_field(field_d, orig_name, sr)
        else:
            fd = pd.DataFrame({'Season': sr})
            field_full = field_d.set_index('Season').reindex(sr, fill_value=0).reset_index() if not field_d.empty else fd.assign(
                catches=0)
        if not field_full.empty:
            print("\nFielding Statistics:")
            print(field_full[['Season', 'catches']].to_string(index=False))
        else:
            print("\nNo fielding stats available.")
        pom_d = p_data[p_data['pom'] > 0][['Season', 'pom']]
        pom_d = pom_d.sort_values('Season')
        if 4 in ch:
            figs[4], pom_full = plot_pom(pom_d, orig_name, sr)
        else:
            fd = pd.DataFrame({'Season': sr})
            pom_full = pom_d.set_index('Season').reindex(sr, fill_value=0).reset_index() if not pom_d.empty else fd.assign(
                pom=0)
        if not pom_full.empty:
            print("\nP.O.M Awards:")
            print(pom_full[['Season', 'pom']].to_string(index=False))
        else:
            print("\nNo P.O.M stats available.")
        if 5 in ch:
            figs[5], ovr, ph = bat_an(name)
            if ovr is not None:
                print("\nOver-by-Over Statistics:")
                print(ovr[['over_num', 'runs', 'balls', 'dismissals', 'avg', 'sr']].to_string(index=False))
            else:
                print(f"\nNo over-by-over stats for {orig_name}.")
        else:
            ovr, ph = None, None
            print(f"\nOver-by-Over stats not requested for {orig_name}.")
        if figs:
            print("\nDisplaying selected plots in separate windows.")
            for num in ch:
                if num in figs and figs[num] is not None:
                    plt.figure(figs[num].number)
                    plt.show()
        inp = input("\nType 'exit' to quit or press Enter to select another player: ").strip().lower()
        if inp == 'exit':
            print("Exiting program.")
            sys.exit(0)
if __name__ == "__main__":
    main()