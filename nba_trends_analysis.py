"""NBA Trends - Data Analysis (refactored)
Refactored analysis with functions, validation, better plotting, and statistical tests.
"""

from pathlib import Path
from typing import Tuple, Optional
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
    """Compute Cohen's d for two independent samples.

    Cohen's d is calculated using pooled standard deviation. Returns NaN if
    samples are too small to estimate an effect size reliably.
    """
    na = a.dropna(); nb = b.dropna()
    n1, n2 = len(na), len(nb)
    if n1 < 2 or n2 < 2:
        return math.nan
    s1, s2 = na.std(ddof=1), nb.std(ddof=1)
    pooled_sd = math.sqrt(((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return (na.mean() - nb.mean()) / pooled_sd


def _interpret_cohens_d(d: float) -> str:
    """Return a human-friendly label for the magnitude of Cohen's d.

    Thresholds (conventional): 0.2 small, 0.5 medium, 0.8 large.
    """
    if math.isnan(d):
        return "insufficient data"
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def compare_team_points(df: pd.DataFrame, year: int, team1: str, team2: str, bins: int = 20, save_dir: Optional[Path] = None) -> dict:
    """Compare points between two teams in a given year.

    Prints descriptive statistics and a statistical test, plots distributions,
    optionally saves the plot to `save_dir`, and returns a dict of numeric
    results for summary/interpretation.
    """
    # Safely extract point series for each team
    a = team_points(df, year, team1)
    b = team_points(df, year, team2)

    print(f"\n{year} - Comparing {team1} vs {team2}:")
    print(f"  {team1}: n={len(a)}, mean={a.mean():.2f}, std={a.std(ddof=1):.2f}")
    print(f"  {team2}: n={len(b)}, mean={b.mean():.2f}, std={b.std(ddof=1):.2f}")

    # Difference in means and Welch's t-test (does not assume equal variances)
    mean_diff = a.mean() - b.mean()
    t_stat, pval = ttest_ind(a, b, equal_var=False, nan_policy='omit')
    d = cohens_d(a, b)

    # More informative print statements with interpretation cues
    print(f"  Difference in Means ({team1} - {team2}): {mean_diff:.2f}")
    print(f"  Welch t-test: t={t_stat:.3f}, p-value={pval:.3e}")
    if pval < 0.05:
        print("    -> p < 0.05: difference is statistically significant at alpha=0.05")
    else:
        print("    -> p >= 0.05: no evidence to reject the null of equal means")
    print(f"  Cohen's d = {d:.3f} ({_interpret_cohens_d(d)})")

    # Plot densities to visualize distributions and overlap
    plt.figure(figsize=(8, 5))
    sns.histplot(a, color='tab:blue', label=team1, stat='density', kde=True, bins=bins, alpha=0.6)
    sns.histplot(b, color='tab:orange', label=team2, stat='density', kde=True, bins=bins, alpha=0.6)
    plt.legend()
    plt.title(f"{year}: {team1} vs {team2} Points (density)")
    plt.xlabel('Points')
    plt.tight_layout()

    fig_path = None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig_path = save_dir / f"{year}_{team1}_vs_{team2}_points.png"
        plt.savefig(fig_path, dpi=150)
        print(f"  Saved density plot to: {fig_path}")

    plt.show()
    plt.close()

    # Return results for later summary
    return {
        'year': year,
        'team1': team1,
        'team2': team2,
        'n1': len(a),
        'n2': len(b),
        'mean_diff': mean_diff,
        't_stat': t_stat,
        'p_value': pval,
        'cohens_d': d,
        'cohens_d_label': _interpret_cohens_d(d),
        'fig_path': str(fig_path) if fig_path is not None else None,
    }


def boxplot_by_franchise(df: pd.DataFrame, year: int, save_dir: Optional[Path] = None) -> Optional[str]:
    """Create a boxplot of points by franchise for a given year.

    Optionally save the plot to `save_dir` and return the saved path.
    """
    df_y = df[df.year_id == year]
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_y, y='pts', x='fran_id')
    plt.title(f"{year} Points by Franchise")
    plt.xticks(rotation=45)
    plt.tight_layout()

    fig_path = None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig_path = save_dir / f"{year}_points_by_franchise_boxplot.png"
        plt.savefig(fig_path, dpi=150)
        print(f"  Saved boxplot to: {fig_path}")

    plt.show()
    plt.close()
    return str(fig_path) if fig_path is not None else None


def contingency_analysis(df: pd.DataFrame, year: int) -> dict:
    """Perform contingency analysis of game location vs result.

    Prints the count and row-normalized proportions (win rate per location),
    runs a chi-square test of independence, and returns the test statistics.
    """
    df_y = df[df.year_id == year]
    table = pd.crosstab(df_y.game_location, df_y.game_result)

    print(f"\nContingency Table (counts) for {year}:")
    print(table)

    # Proportions by location show win/loss rates for each location (rows sum to 1)
    proportions = table.div(table.sum(axis=1), axis=0)
    print(f"\nProportions (win rate by location) for {year}:")
    print(proportions)

    chi2, p, dof, expected = chi2_contingency(table)
    print(f"\nChi-square test of independence: chi2={chi2:.3f}, p-value={p:.3e}, dof={dof}")
    if p < 0.05:
        print("  -> p < 0.05: evidence suggests game location and game result are not independent (association present)")
    else:
        print("  -> p >= 0.05: no evidence of association between location and result")

    expected_df = pd.DataFrame(expected, index=table.index, columns=table.columns)
    print("Expected frequencies under independence:")
    print(expected_df)

    return {
        'year': year,
        'table': table,
        'proportions': proportions,
        'chi2': chi2,
        'p_value': p,
        'dof': dof,
        'expected': expected_df,
    }


def _interpret_corr(r: float) -> str:
    """Simple interpretation of correlation magnitude."""
    ar = abs(r)
    if ar < 0.1:
        return "very weak"
    if ar < 0.3:
        return "weak"
    if ar < 0.5:
        return "moderate"
    return "strong"


def correlation_analysis(df: pd.DataFrame, year: int, save_dir: Optional[Path] = None) -> dict:
    """Compute and display correlation between forecast and point differential.

    Returns correlation coefficient and p-value for later summary. Optionally
    saves the regression scatter plot to `save_dir`.
    """
    df_y = df[df.year_id == year][['forecast', 'point_diff']].dropna()
    if len(df_y) < 2:
        print("Not enough data for correlation")
        return {'year': year, 'n': len(df_y)}

    # Covariance gives joint variability; Pearson's r gives standardized correlation
    cov = np.cov(df_y['forecast'], df_y['point_diff'])
    corr, pval = pearsonr(df_y['forecast'], df_y['point_diff'])

    print(f"\nCorrelation analysis for {year} (n={len(df_y)}):")
    print("  Covariance matrix:\n", cov)
    print(f"  Pearson r = {corr:.3f}, p-value = {pval:.3e} ({_interpret_corr(corr)})")

    # Scatter with regression line to visualize relationship
    plt.figure(figsize=(7, 5))
    sns.regplot(data=df_y, x='forecast', y='point_diff', scatter_kws={'alpha':0.6})
    plt.title(f"{year}: Forecast vs Point Differential (r={corr:.2f})")
    plt.xlabel('Forecasted Win Probability')
    plt.ylabel('Point Differential')
    plt.tight_layout()

    fig_path = None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig_path = save_dir / f"{year}_forecast_vs_pointdiff.png"
        plt.savefig(fig_path, dpi=150)
        print(f"  Saved correlation plot to: {fig_path}")

    plt.show()
    plt.close()

    return {
        'year': year,
        'n': len(df_y),
        'covariance': cov,
        'pearson_r': corr,
        'p_value': pval,
        'corr_label': _interpret_corr(corr),
        'fig_path': str(fig_path) if fig_path is not None else None,
    }


def interpretation_summary(results_by_year: dict):
    """Print a concise, human-readable interpretation of all results.

    This aggregates the numeric results returned by the analysis functions
    and provides plain-language takeaways for quick reporting.
    """
    print("\n=== Final Interpretation Summary ===\n")
    for year, res in results_by_year.items():
        print(f"Year: {year}")
        # Team comparison summary (if present)
        tp = res.get('team_comparison')
        if tp:
            print(f"  {tp['team1']} vs {tp['team2']}: mean difference = {tp['mean_diff']:.2f}")
            print(f"    p = {tp['p_value']:.3e} ({'statistically significant' if tp['p_value'] < 0.05 else 'not significant'})")
            print(f"    Cohen's d = {tp['cohens_d']:.3f} ({tp['cohens_d_label']})")

        # Contingency summary (if present)
        cont = res.get('contingency')
        if cont:
            p = cont['p_value']
            print(f"  Game location vs result: chi2 p = {p:.3e} ({'association present' if p < 0.05 else 'no association'})")

        # Correlation summary (if present)
        corr = res.get('correlation')
        if corr:
            print(f"  Forecast vs point_diff: r = {corr['pearson_r']:.3f} ({corr['corr_label']}), p = {corr['p_value']:.3e}")

        print("")

    # Overall takeaways (simple heuristics)
    print("Takeaways:")
    # Example inference: if any year has a large effect size for points difference, flag it
    for year, res in results_by_year.items():
        tp = res.get('team_comparison')
        if tp and tp['cohens_d_label'] == 'large' and tp['p_value'] < 0.05:
            print(f"  - {year}: Large, significant scoring difference between {tp['team1']} and {tp['team2']} (d={tp['cohens_d']:.2f}).")
    print("  - Consider reporting these results alongside plots and exact tables in any write-up.")
    print("\nNote: 'statistically significant' does not necessarily imply practical importance; consider effect sizes and context.")


def main(data_path: Path = DATA_FILENAME):
    try:
        nba = load_data(data_path)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        return

    # Quick preview
    print("Data preview:\n", nba.head())

    results = {}

    # Create directory for figures and analyze seasons
    fig_dir = Path(__file__).parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for year in (2010, 2014):
        print(f"\n--- Analysis for {year} ---")
        results[year] = {}

        # Team comparisons (returns a dict)
        tp = compare_team_points(nba, year, 'Knicks', 'Nets', save_dir=fig_dir)
        results[year]['team_comparison'] = tp

        # Boxplot (saved to figures)
        bp_path = boxplot_by_franchise(nba, year, save_dir=fig_dir)
        results[year]['boxplot_path'] = bp_path

        # Contingency
        cont = contingency_analysis(nba, year)
        results[year]['contingency'] = cont

        # Correlation (saved to figures)
        corr = correlation_analysis(nba, year, save_dir=fig_dir)
        results[year]['correlation'] = corr

    # Print final interpretation/summary based on collected metrics
    interpretation_summary(results)


if __name__ == "__main__":
    main()
