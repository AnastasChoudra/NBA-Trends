"""NBA Trends - Data Analysis (refactored)
Refactored analysis with functions, validation, better plotting, and statistical tests.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, chi2_contingency, ttest_ind
import math
import sys

sns.set(style="whitegrid", context="talk")
np.set_printoptions(suppress=True, precision=2)

DATA_FILENAME = Path(__file__).parent / "nba_games.csv"


def load_data(path: Path) -> pd.DataFrame:
    """Load CSV into DataFrame and basic validation."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    # Basic required columns
    required = {"year_id", "fran_id", "pts", "game_location", "game_result", "forecast", "point_diff"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")
    return df


def team_points(df: pd.DataFrame, year: int, team: str) -> pd.Series:
    """Return points for a given team and year."""
    mask = (df["year_id"] == year) & (df["fran_id"] == team)
    return df.loc[mask, "pts"].dropna()


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    """Compute Cohen's d for two independent samples."""
    na = a.dropna(); nb = b.dropna()
    n1, n2 = len(na), len(nb)
    if n1 < 2 or n2 < 2:
        return math.nan
    s1, s2 = na.std(ddof=1), nb.std(ddof=1)
    pooled_sd = math.sqrt(((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return (na.mean() - nb.mean()) / pooled_sd


def compare_team_points(df: pd.DataFrame, year: int, team1: str, team2: str, bins: int = 20):
    a = team_points(df, year, team1)
    b = team_points(df, year, team2)
    print(f"\n{year} {team1} (n={len(a)}) mean={a.mean():.2f} std={a.std(ddof=1):.2f}")
    print(f"{year} {team2} (n={len(b)}) mean={b.mean():.2f} std={b.std(ddof=1):.2f}")

    # Difference in means and t-test
    mean_diff = a.mean() - b.mean()
    t_stat, pval = ttest_ind(a, b, equal_var=False, nan_policy='omit')
    d = cohens_d(a, b)
    print(f"Difference in Means ({team1} - {team2}): {mean_diff:.2f}")
    print(f"Welch t-test: t={t_stat:.3f}, p={pval:.3e}, Cohen's d={d:.3f}")

    # Plot
    plt.figure(figsize=(8, 5))
    sns.histplot(a, color='tab:blue', label=team1, stat='density', kde=True, bins=bins, alpha=0.6)
    sns.histplot(b, color='tab:orange', label=team2, stat='density', kde=True, bins=bins, alpha=0.6)
    plt.legend()
    plt.title(f"{year}: {team1} vs {team2} Points (density)")
    plt.xlabel('Points')
    plt.tight_layout()
    plt.show()


def boxplot_by_franchise(df: pd.DataFrame, year: int):
    df_y = df[df.year_id == year]
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_y, y='pts', x='fran_id')
    plt.title(f"{year} Points by Franchise")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def contingency_analysis(df: pd.DataFrame, year: int):
    df_y = df[df.year_id == year]
    table = pd.crosstab(df_y.game_location, df_y.game_result)
    print(f"\nContingency Table (counts) for {year}:\n", table)
    # Normalize to proportions by row (location)
    proportions = table.div(table.sum(axis=1), axis=0)
    print(f"\nProportions (by location) for {year}:\n", proportions)
    chi2, p, dof, expected = chi2_contingency(table)
    print(f"\nChi2 test: chi2={chi2:.3f}, p={p:.3e}, dof={dof}")
    print("Expected frequencies:\n", pd.DataFrame(expected, index=table.index, columns=table.columns))


def correlation_analysis(df: pd.DataFrame, year: int):
    df_y = df[df.year_id == year][['forecast', 'point_diff']].dropna()
    if len(df_y) < 2:
        print("Not enough data for correlation")
        return
    cov = np.cov(df_y['forecast'], df_y['point_diff'])
    corr, pval = pearsonr(df_y['forecast'], df_y['point_diff'])
    print(f"\nCovariance matrix for {year}:\n{cov}")
    print(f"Pearson r={corr:.3f}, p={pval:.3e}")

    plt.figure(figsize=(7, 5))
    sns.regplot(data=df_y, x='forecast', y='point_diff', scatter_kws={'alpha':0.6})
    plt.title(f"{year}: Forecast vs Point Differential (r={corr:.2f})")
    plt.tight_layout()
    plt.show()


def main(data_path: Path = DATA_FILENAME):
    try:
        nba = load_data(data_path)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        return

    # Quick preview
    print("Data preview:\n", nba.head())

    # Analyze seasons
    for year in (2010, 2014):
        print(f"\n--- Analysis for {year} ---")
        compare_team_points(nba, year, 'Knicks', 'Nets')
        boxplot_by_franchise(nba, year)
        contingency_analysis(nba, year)
        correlation_analysis(nba, year)


if __name__ == "__main__":
    main()
