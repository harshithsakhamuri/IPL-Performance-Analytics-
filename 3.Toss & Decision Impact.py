import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency

def toss_decision_analysis():
    m_file = r"C:\Users\sakha\Downloads\PythonProject\IPL_Matches_2008_2022.csv"
    b_file = r"C:\Users\sakha\Downloads\PythonProject\IPL_Ball_by_Ball_2008_2022.csv"
    try:
        m_df = pd.read_csv(m_file, low_memory=False)
        b_df = pd.read_csv(b_file, low_memory=False)
    except Exception as e:
        print(f"Error loading files: {e}")
        return None, None
    id_col = next((col for col in ['MatchID', 'ID', 'match_id', 'MatchId', 'id'] if col in m_df.columns and col in b_df.columns), None)
    if not id_col:
        print("Error: No common match identifier found.")
        return None, None
    v_m = m_df[~m_df['WinningTeam'].isin(['No Result', 'Abandoned'])].copy()
    v_m['TossWin_MatchWin'] = v_m['TossWinner'] == v_m['WinningTeam']
    v_m['TossLose_MatchWin'] = v_m['TossWinner'] != v_m['WinningTeam']
    v_m['WinType'] = 'Comfortable'
    v_m.loc[(v_m['WonBy'] == 'Runs') & (v_m['Margin'] < 15), 'WinType'] = 'Close'
    for idx, row in v_m[v_m['WonBy'] == 'Wickets'].iterrows():
        i_data = b_df[(b_df[id_col] == row[id_col]) & (b_df['BattingTeam'] == row['WinningTeam'])]
        if not i_data.empty and i_data['overs'].max() >= 19:
            v_m.loc[idx, 'WinType'] = 'Close'
    t_a = []
    for cat, cond in [
        ('tosswin&bat', v_m['TossWin_MatchWin'] & (v_m['TossDecision'] == 'bat')),
        ('tosswin&bowl', v_m['TossWin_MatchWin'] & (v_m['TossDecision'] == 'field')),
        ('tosslose&bat', v_m['TossLose_MatchWin'] & (v_m['TossDecision'] == 'field')),
        ('tosslose&bowl', v_m['TossLose_MatchWin'] & (v_m['TossDecision'] == 'bat'))
    ]:
        c_m = v_m[cond]
        t_a.append({
            'Category': cat,
            'Comfortable': len(c_m[c_m['WinType'] == 'Comfortable']),
            'Close': len(c_m[c_m['WinType'] == 'Close']),
            'TotalWins': len(c_m)
        })
    t_df = pd.DataFrame(t_a)
    t_df['Category%'] = (t_df['TotalWins'] / len(v_m) * 100).round(1)
    t_df['Comfortable%'] = (t_df['Comfortable'] / t_df['TotalWins'] * 100).round(1)
    t_df['Close%'] = (t_df['Close'] / t_df['TotalWins'] * 100).round(1)
    print(t_df[['Category', 'Comfortable', 'Close', 'TotalWins', 'Category%', 'Comfortable%', 'Close%']])
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = {'tosswin&bat': '#1f77b4', 'tosswin&bowl': '#90EE90', 'tosslose&bat': '#d62728', 'tosslose&bowl': '#ff7f0e'}
    w = 0.35
    x = range(len(t_df))
    ax.bar([i - w/2 for i in x], t_df['Comfortable'], width=w, color=[colors[c] for c in t_df['Category']], label='Comfortable Wins')
    ax.bar([i + w/2 for i in x], t_df['Close'], width=w, color=[colors[c] for c in t_df['Category']], hatch='//', label='Close Wins')
    for i, r in t_df.iterrows():
        ax.text(i, r['TotalWins'] + 5, f"{r['Category%']}%", ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        ax.text(i - w/2, r['Comfortable'] + 2, f"{r['Comfortable']}\n({r['Comfortable%']}%)", ha='center', fontsize=9)
        ax.text(i + w/2, r['Close'] + 2, f"{r['Close']}\n({r['Close%']}%)", ha='center', fontsize=9)
    ax.set_title('Toss Decision Impact', fontsize=14)
    ax.set_xlabel('Toss Decision Outcome', fontsize=12)
    ax.set_ylabel('Matches Won', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(["Toss Win & Bat", "Toss Win & Field", "Toss Lose & Field", "Toss Lose & Bat"], rotation=15, ha='right')
    ax.legend(title='Win Type')
    ax.set_ylim(0, t_df['TotalWins'].max() * 1.3)
    plt.tight_layout()
    try:
        t_df.to_csv('toss_decision_results.csv', index=False)
        print("Results saved as 'toss_decision_results.csv'")
    except Exception as e:
        print(f"Error saving results: {e}")
    h_data = pd.crosstab(v_m['TossDecision'].map({'bat': 'Bat First', 'field': 'Field First'}),
                         v_m['TossWin_MatchWin'].map({True: 'Won Match', False: 'Lost Match'}))
    h_data = h_data[['Won Match', 'Lost Match']]
    chi2, p, _, _ = chi2_contingency(h_data)
    n = h_data.sum().sum()
    phi = np.sqrt(chi2/n)
    a_m = h_data.copy().astype(str)
    for i in range(h_data.shape[0]):
        for j in range(h_data.shape[1]):
            o = h_data.iloc[i,j]
            e = (h_data.sum(axis=0)[j] * h_data.sum(axis=1)[i]) / n
            a_m.iloc[i,j] = f"{o}\n{((o - e) / np.sqrt(e):.2f)-1}"
    plt.figure(figsize=(8, 6))
    sns.heatmap(h_data, annot=a_m, fmt='', cmap='YlGnBu', cbar_kws={'label': 'Matches'}, linewidths=0.5, linecolor='gray')
    plt.title('Toss Decision vs. Match Result', pad=20)
    plt.xlabel('Match Result')
    plt.ylabel('Toss Decision')
    plt.tight_layout()
    try:
        plt.savefig("toss_decision_correlation.png", dpi=300)
        print("Heatmap saved to: toss_decision_correlation.png")
    except Exception as e:
        print(f"Error saving heatmap: {e}")
    return fig, t_df

if __name__ == "__main__":
    fig, r = toss_decision_analysis()
    if fig:
        plt.show()
        plt.show()