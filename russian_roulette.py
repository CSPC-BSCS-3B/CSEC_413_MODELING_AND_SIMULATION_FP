"""
Russian Roulette Casino Game Simulation
========================================
CSEC 413 - Modeling and Simulation Final Project
Option 2: Stochastic Game Simulation

This project models Russian Roulette as a casino betting game where players
bet on their survival. We implement:
1. Fair Game Model - Equal expected value for player and house
2. Tweaked Game Models - House edge through various techniques
3. Monte Carlo Simulation - 10,000+ plays per model
4. Exploratory Data Analysis (EDA)
5. Comparison of fair vs tweaked outcomes

Author: CSPC-BSCS-3B
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict
import random
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# GAME MODELS
# ============================================================================

class RussianRouletteGame:
    """
    Base class for Russian Roulette Casino Game.
    
    Game Rules:
    - Revolver has 6 chambers, 1 bullet
    - Player bets money and "pulls the trigger"
    - If empty chamber (CLICK) -> Player WINS, gets payout
    - If bullet fires (BANG) -> Player LOSES bet
    
    Fair Probability:
    - P(Win) = 5/6 = 83.33%
    - P(Lose) = 1/6 = 16.67%
    
    Fair Payout for zero expected value:
    - If player wins, they get bet * (1/5) = 0.2x profit
    - Expected Value = P(Win) * Payout - P(Lose) * Bet
    - EV = (5/6) * 0.2 - (1/6) * 1 = 0.1667 - 0.1667 = 0
    """
    
    def __init__(self, chambers: int = 6, bullets: int = 1, 
                 payout_multiplier: float = 0.2, is_fair: bool = True,
                 weighted_probs: List[float] = None):
        """
        Initialize the Russian Roulette game.
        
        Args:
            chambers: Number of chambers in revolver (default 6)
            bullets: Number of bullets (default 1)
            payout_multiplier: Multiplier for winnings (default 0.2 for fair game)
            is_fair: If True, use fair probabilities
            weighted_probs: Custom probabilities [P(empty), P(bullet)] for tweaked games
        """
        self.chambers = chambers
        self.bullets = bullets
        self.payout_multiplier = payout_multiplier
        self.is_fair = is_fair
        
        # Calculate probabilities
        if weighted_probs is not None:
            self.p_win = weighted_probs[0]  # P(empty/click/win)
            self.p_lose = weighted_probs[1]  # P(bullet/bang/lose)
        else:
            self.p_win = (chambers - bullets) / chambers  # Fair: 5/6
            self.p_lose = bullets / chambers  # Fair: 1/6
        
        # Calculate theoretical expected value per unit bet
        self.expected_value = self.p_win * self.payout_multiplier - self.p_lose
        self.house_edge = -self.expected_value  # House edge is negative of player EV
    
    def play_single_round(self, bet: float = 1.0) -> Tuple[bool, float]:
        """
        Play a single round of Russian Roulette.
        
        Args:
            bet: Amount wagered
            
        Returns:
            Tuple of (won: bool, net_result: float)
        """
        # Determine outcome based on probability
        outcome = np.random.choice(['win', 'lose'], p=[self.p_win, self.p_lose])
        
        if outcome == 'win':
            net_result = bet * self.payout_multiplier  # Player wins
            return True, net_result
        else:
            net_result = -bet  # Player loses bet
            return False, net_result
    
    def get_game_info(self) -> Dict:
        """Return game parameters and statistics."""
        return {
            'chambers': self.chambers,
            'bullets': self.bullets,
            'p_win': self.p_win,
            'p_lose': self.p_lose,
            'payout_multiplier': self.payout_multiplier,
            'expected_value': self.expected_value,
            'house_edge': self.house_edge,
            'house_edge_percent': self.house_edge * 100
        }


# ============================================================================
# MONTE CARLO SIMULATION ENGINE
# ============================================================================

class MonteCarloSimulator:
    """
    Monte Carlo Simulation Engine for Russian Roulette games.
    
    Runs thousands of simulations to analyze:
    - Player outcomes (wins/losses)
    - House profit/loss
    - Statistical distributions
    """
    
    def __init__(self, game: RussianRouletteGame):
        """Initialize simulator with a game model."""
        self.game = game
        self.results = None
    
    def run_simulation(self, num_plays: int = 10000, bet_per_play: float = 100.0) -> pd.DataFrame:
        """
        Run Monte Carlo simulation.
        
        Args:
            num_plays: Number of games to simulate (default 10,000)
            bet_per_play: Bet amount per game
            
        Returns:
            DataFrame with simulation results
        """
        results = []
        cumulative_player_balance = 0
        cumulative_house_balance = 0
        
        for play_num in range(1, num_plays + 1):
            won, net_result = self.game.play_single_round(bet_per_play)
            
            cumulative_player_balance += net_result
            cumulative_house_balance -= net_result  # House is opposite of player
            
            results.append({
                'play_number': play_num,
                'won': won,
                'bet': bet_per_play,
                'net_result': net_result,
                'cumulative_player_balance': cumulative_player_balance,
                'cumulative_house_balance': cumulative_house_balance
            })
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def get_statistics(self) -> Dict:
        """Calculate summary statistics from simulation results."""
        if self.results is None:
            raise ValueError("Run simulation first!")
        
        total_plays = len(self.results)
        wins = self.results['won'].sum()
        losses = total_plays - wins
        
        return {
            'total_plays': total_plays,
            'wins': wins,
            'losses': losses,
            'win_rate': wins / total_plays,
            'loss_rate': losses / total_plays,
            'total_wagered': self.results['bet'].sum(),
            'player_final_balance': self.results['cumulative_player_balance'].iloc[-1],
            'house_final_balance': self.results['cumulative_house_balance'].iloc[-1],
            'player_mean_result': self.results['net_result'].mean(),
            'player_std_result': self.results['net_result'].std(),
            'player_max_win': self.results['net_result'].max(),
            'player_max_loss': self.results['net_result'].min(),
            'empirical_house_edge': -self.results['net_result'].mean() / self.results['bet'].iloc[0],
            'theoretical_house_edge': self.game.house_edge
        }


# ============================================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

class GameAnalyzer:
    """
    Performs Exploratory Data Analysis on simulation results.
    Generates visualizations and statistical comparisons.
    """
    
    def __init__(self):
        """Initialize analyzer with storage for multiple game results."""
        self.simulations = {}
        self.statistics = {}
    
    def add_simulation(self, name: str, game: RussianRouletteGame, 
                       results: pd.DataFrame, stats: Dict):
        """Add simulation results for analysis."""
        self.simulations[name] = {
            'game': game,
            'results': results,
            'stats': stats
        }
        self.statistics[name] = stats
    
    def print_game_comparison(self):
        """Print comparison table of all game models."""
        print("\n" + "=" * 80)
        print("GAME MODEL COMPARISON")
        print("=" * 80)
        
        comparison_data = []
        for name, sim in self.simulations.items():
            game_info = sim['game'].get_game_info()
            stats = sim['stats']
            comparison_data.append({
                'Model': name,
                'P(Win)': f"{game_info['p_win']:.4f}",
                'P(Lose)': f"{game_info['p_lose']:.4f}",
                'Payout': f"{game_info['payout_multiplier']:.4f}",
                'Theoretical EV': f"{game_info['expected_value']:.4f}",
                'Empirical Win Rate': f"{stats['win_rate']:.4f}",
                'Empirical House Edge': f"{stats['empirical_house_edge']*100:.2f}%",
                'House Profit': f"₱{stats['house_final_balance']:,.2f}"
            })
        
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        print()
    
    def print_detailed_statistics(self):
        """Print detailed statistics for each simulation."""
        for name, sim in self.simulations.items():
            stats = sim['stats']
            game_info = sim['game'].get_game_info()
            
            print(f"\n{'=' * 60}")
            print(f"DETAILED STATISTICS: {name}")
            print(f"{'=' * 60}")
            print(f"\nGame Parameters:")
            print(f"  - Chambers: {game_info['chambers']}")
            print(f"  - Bullets: {game_info['bullets']}")
            print(f"  - P(Win/Click): {game_info['p_win']:.4f} ({game_info['p_win']*100:.2f}%)")
            print(f"  - P(Lose/Bang): {game_info['p_lose']:.4f} ({game_info['p_lose']*100:.2f}%)")
            print(f"  - Payout Multiplier: {game_info['payout_multiplier']:.4f}")
            print(f"\nTheoretical Values:")
            print(f"  - Expected Value (per ₱1 bet): ₱{game_info['expected_value']:.4f}")
            print(f"  - House Edge: {game_info['house_edge_percent']:.2f}%")
            print(f"\nSimulation Results ({stats['total_plays']:,} plays):")
            print(f"  - Wins: {stats['wins']:,} ({stats['win_rate']*100:.2f}%)")
            print(f"  - Losses: {stats['losses']:,} ({stats['loss_rate']*100:.2f}%)")
            print(f"  - Total Wagered: ₱{stats['total_wagered']:,.2f}")
            print(f"  - Player Final Balance: ₱{stats['player_final_balance']:,.2f}")
            print(f"  - House Final Balance: ₱{stats['house_final_balance']:,.2f}")
            print(f"\nPlayer Statistics:")
            print(f"  - Mean Result per Play: ₱{stats['player_mean_result']:.2f}")
            print(f"  - Std Dev: ₱{stats['player_std_result']:.2f}")
            print(f"  - Max Win: ₱{stats['player_max_win']:.2f}")
            print(f"  - Max Loss: ₱{stats['player_max_loss']:.2f}")
            print(f"\nEmpirical vs Theoretical:")
            print(f"  - Empirical House Edge: {stats['empirical_house_edge']*100:.2f}%")
            print(f"  - Theoretical House Edge: {stats['theoretical_house_edge']*100:.2f}%")
            print(f"  - Difference: {abs(stats['empirical_house_edge'] - stats['theoretical_house_edge'])*100:.4f}%")
    
    def plot_cumulative_balances(self, figsize=(14, 8)):
        """Plot cumulative player balances over time for all models."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Russian Roulette Monte Carlo Simulation Results', fontsize=14, fontweight='bold')
        
        colors = ['green', 'red', 'orange', 'purple']
        
        # Plot 1: Player cumulative balance
        ax1 = axes[0, 0]
        for idx, (name, sim) in enumerate(self.simulations.items()):
            results = sim['results']
            ax1.plot(results['play_number'], results['cumulative_player_balance'], 
                    label=name, color=colors[idx % len(colors)], alpha=0.8)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Number of Plays')
        ax1.set_ylabel('Player Cumulative Balance (₱)')
        ax1.set_title('Player Balance Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: House cumulative balance
        ax2 = axes[0, 1]
        for idx, (name, sim) in enumerate(self.simulations.items()):
            results = sim['results']
            ax2.plot(results['play_number'], results['cumulative_house_balance'], 
                    label=name, color=colors[idx % len(colors)], alpha=0.8)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Number of Plays')
        ax2.set_ylabel('House Cumulative Balance (₱)')
        ax2.set_title('House Profit Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Win rate comparison
        ax3 = axes[1, 0]
        names = list(self.simulations.keys())
        win_rates = [self.simulations[n]['stats']['win_rate'] * 100 for n in names]
        theoretical_rates = [self.simulations[n]['game'].p_win * 100 for n in names]
        
        x = np.arange(len(names))
        width = 0.35
        bars1 = ax3.bar(x - width/2, win_rates, width, label='Empirical', color='steelblue')
        bars2 = ax3.bar(x + width/2, theoretical_rates, width, label='Theoretical', color='lightcoral')
        ax3.set_xlabel('Game Model')
        ax3.set_ylabel('Win Rate (%)')
        ax3.set_title('Win Rate: Empirical vs Theoretical')
        ax3.set_xticks(x)
        ax3.set_xticklabels(names, rotation=15)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax3.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        
        # Plot 4: House edge comparison
        ax4 = axes[1, 1]
        house_edges = [self.simulations[n]['stats']['empirical_house_edge'] * 100 for n in names]
        theoretical_edges = [self.simulations[n]['game'].house_edge * 100 for n in names]
        
        bars3 = ax4.bar(x - width/2, house_edges, width, label='Empirical', color='darkgreen')
        bars4 = ax4.bar(x + width/2, theoretical_edges, width, label='Theoretical', color='lightgreen')
        ax4.set_xlabel('Game Model')
        ax4.set_ylabel('House Edge (%)')
        ax4.set_title('House Edge: Empirical vs Theoretical')
        ax4.set_xticks(x)
        ax4.set_xticklabels(names, rotation=15)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        for bar in bars3:
            height = bar.get_height()
            ax4.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('simulation_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("\n📊 Plot saved as 'simulation_results.png'")
    
    def plot_outcome_distributions(self, figsize=(14, 6)):
        """Plot distribution of outcomes for each model."""
        fig, axes = plt.subplots(1, len(self.simulations), figsize=figsize)
        fig.suptitle('Distribution of Single Play Outcomes', fontsize=14, fontweight='bold')
        
        if len(self.simulations) == 1:
            axes = [axes]
        
        for idx, (name, sim) in enumerate(self.simulations.items()):
            ax = axes[idx]
            results = sim['results']
            
            # Create histogram of net results
            ax.hist(results['net_result'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            ax.axvline(x=results['net_result'].mean(), color='red', linestyle='--', 
                      label=f'Mean: ₱{results["net_result"].mean():.2f}')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            ax.set_xlabel('Net Result (₱)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{name}\nMean: ₱{results["net_result"].mean():.2f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outcome_distributions.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("📊 Plot saved as 'outcome_distributions.png'")
    
    def plot_house_profit_comparison(self, figsize=(10, 6)):
        """Bar chart comparing final house profits."""
        fig, ax = plt.subplots(figsize=figsize)
        
        names = list(self.simulations.keys())
        profits = [self.simulations[n]['stats']['house_final_balance'] for n in names]
        
        colors = ['green' if p > 0 else 'red' for p in profits]
        bars = ax.bar(names, profits, color=colors, edgecolor='black', alpha=0.8)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Game Model')
        ax.set_ylabel('House Final Balance (₱)')
        ax.set_title('Final House Profit Comparison (After 10,000 Plays)\n₱100 Bet Per Play', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, profit in zip(bars, profits):
            height = bar.get_height()
            ax.annotate(f'₱{profit:,.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 5 if height >= 0 else -15), textcoords="offset points",
                       ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('house_profit_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("📊 Plot saved as 'house_profit_comparison.png'")
    
    def plot_detectability_analysis(self, figsize=(14, 10)):
        """
        Demonstrate WHY modified payouts go unnoticed.
        Shows how variance masks the house edge in short sessions.
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Why Players Don\'t Notice the Rigged Gun (Weighted Probabilities)', 
                    fontsize=14, fontweight='bold')
        
        # Get fair and tweaked games
        fair_results = self.simulations['1. Fair Game']['results']
        tweaked_results = self.simulations['2. Weighted Probs (Rigged)']['results']
        
        # Plot 1: Single session comparison (10 plays)
        ax1 = axes[0, 0]
        session_sizes = [10, 50, 100, 500, 1000, 5000, 10000]
        fair_means = [fair_results['net_result'][:n].mean() for n in session_sizes]
        tweaked_means = [tweaked_results['net_result'][:n].mean() for n in session_sizes]
        
        ax1.plot(session_sizes, fair_means, 'go-', label='Fair Game', linewidth=2, markersize=8)
        ax1.plot(session_sizes, tweaked_means, 'rs-', label='Rigged Gun', linewidth=2, markersize=8)
        ax1.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Fair EV (₱0)')
        ax1.axhline(y=-4.00, color='red', linestyle='--', alpha=0.5, label='Rigged EV (-₱4.00)')
        ax1.set_xlabel('Number of Plays in Session')
        ax1.set_ylabel('Average Result per Play (₱)')
        ax1.set_title('Average Result Converges to True EV Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # Plot 2: Variance comparison - why it's hard to tell
        ax2 = axes[0, 1]
        ax2.hist(fair_results['net_result'], bins=30, alpha=0.6, label='Fair Game', color='green', edgecolor='black')
        ax2.hist(tweaked_results['net_result'], bins=30, alpha=0.6, label='Rigged Gun', color='red', edgecolor='black')
        ax2.axvline(x=fair_results['net_result'].mean(), color='darkgreen', linestyle='-', linewidth=2)
        ax2.axvline(x=tweaked_results['net_result'].mean(), color='darkred', linestyle='-', linewidth=2)
        ax2.set_xlabel('Single Play Result (₱)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Outcome Distribution Looks Almost Identical!')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add annotation showing the distributions overlap
        ax2.annotate('Distributions overlap!\nHard to tell them apart', 
                    xy=(0, 4000), fontsize=10, ha='center',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # Plot 3: Short session simulation - player experiences
        ax3 = axes[1, 0]
        num_sessions = 100
        session_length = 20  # Typical gambling session
        
        fair_session_results = []
        tweaked_session_results = []
        
        for i in range(num_sessions):
            start_idx = i * session_length % (len(fair_results) - session_length)
            fair_session_results.append(fair_results['net_result'][start_idx:start_idx+session_length].sum())
            tweaked_session_results.append(tweaked_results['net_result'][start_idx:start_idx+session_length].sum())
        
        ax3.hist(fair_session_results, bins=20, alpha=0.6, label='Fair Game Sessions', color='green', edgecolor='black')
        ax3.hist(tweaked_session_results, bins=20, alpha=0.6, label='Rigged Gun Sessions', color='red', edgecolor='black')
        ax3.axvline(x=0, color='black', linestyle='--', linewidth=2)
        ax3.set_xlabel(f'Total Profit/Loss in {session_length}-Play Session (₱)')
        ax3.set_ylabel('Number of Sessions')
        ax3.set_title(f'Short Sessions ({session_length} plays): Results Vary Wildly!')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Calculate % of winning sessions
        fair_win_pct = sum(1 for x in fair_session_results if x > 0) / len(fair_session_results) * 100
        tweaked_win_pct = sum(1 for x in tweaked_session_results if x > 0) / len(tweaked_session_results) * 100
        ax3.annotate(f'Fair: {fair_win_pct:.0f}% winning sessions\nRigged: {tweaked_win_pct:.0f}% winning sessions\nHard to notice!', 
                    xy=(0.02, 0.98), xycoords='axes fraction', fontsize=9, va='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        # Plot 4: The revealing truth - long term
        ax4 = axes[1, 1]
        ax4.plot(fair_results['play_number'], fair_results['cumulative_player_balance'], 
                'g-', label='Fair Game', linewidth=2, alpha=0.8)
        ax4.plot(tweaked_results['play_number'], tweaked_results['cumulative_player_balance'], 
                'r-', label='Rigged Gun', linewidth=2, alpha=0.8)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Number of Plays')
        ax4.set_ylabel('Cumulative Player Balance (₱)')
        ax4.set_title('The Truth Revealed Over 10,000 Plays!')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add annotation
        final_fair = fair_results['cumulative_player_balance'].iloc[-1]
        final_tweaked = tweaked_results['cumulative_player_balance'].iloc[-1]
        ax4.annotate(f'Fair: ₱{final_fair:,.0f}\nTweaked: ₱{final_tweaked:,.0f}\nDifference: ₱{final_fair - final_tweaked:,.0f}', 
                    xy=(0.98, 0.02), xycoords='axes fraction', fontsize=10, ha='right', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout()
        plt.savefig('detectability_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("📊 Plot saved as 'detectability_analysis.png'")
        
        # Print explanation
        print("\n" + "=" * 70)
        print("🔍 WHY THE RIGGED GUN GOES UNNOTICED (WEIGHTED PROBABILITIES)")
        print("=" * 70)
        print(f"""
    THE PSYCHOLOGY OF UNDETECTABLE HOUSE EDGES:
    
    1️⃣  SMALL PROBABILITY DIFFERENCE:
        • Fair gun: 16.67% chance of bullet (1 in 6)
        • Rigged gun: 20.00% chance of bullet
        • Difference: Only 3.33% more bullets!
        • Players think: "It's still roughly 1 in 5-6, seems normal"
    
    2️⃣  HIGH VARIANCE MASKS THE EDGE:
        • Single play outcomes: Win +₱20 or Lose -₱100
        • Range of ₱120 swing per play!
        • The 3.33% probability shift is NOISE in this variance
    
    3️⃣  SHORT SESSIONS SHOW RANDOM RESULTS:
        • In {session_length}-play sessions:
        • Fair game: {fair_win_pct:.0f}% of sessions are profitable
        • Rigged game: {tweaked_win_pct:.0f}% of sessions are profitable
        • Almost the same! Players can't tell the difference!
    
    4️⃣  CONFIRMATION BIAS:
        • When players win: "I'm lucky today!"
        • When players lose: "Just bad luck, bullet came up"
        • They never suspect the probabilities are rigged
    
    5️⃣  THE TRUTH ONLY APPEARS OVER TIME:
        • After 10,000 plays:
        • Fair game balance: ₱{final_fair:,.0f}
        • Rigged game balance: ₱{final_tweaked:,.0f}
        • House extracted: ₱{final_fair - final_tweaked:,.0f} extra!
        • But no single player plays 10,000 times!
    
    💡 HOW THE RIGGED GUN WORKS:
        • Instead of real chamber selection, uses weighted random
        • numpy.random.choice(['win','lose'], p=[0.80, 0.20])
        • Looks like a real gun, but probabilities are secretly altered!
    
    🎰 REAL-WORLD PARALLELS (Weighted Probabilities):
        • Loaded dice in street games
        • Rigged wheels in carnival games
        • Digital slot machines with adjusted RNG
        • Players rarely notice or calculate these differences!
    """)


# ============================================================================
# MAIN SIMULATION RUNNER
# ============================================================================

def run_complete_analysis():
    """
    Run the complete Monte Carlo simulation and analysis.
    
    This function:
    1. Creates Fair and Tweaked game models
    2. Runs Monte Carlo simulations (10,000+ plays each)
    3. Performs EDA on results
    4. Compares and evaluates all models
    """
    
    print("=" * 80)
    print("RUSSIAN ROULETTE CASINO GAME - MONTE CARLO SIMULATION")
    print("CSEC 413 - Modeling and Simulation Final Project")
    print("=" * 80)
    
    # ========================================================================
    # STEP 1: CREATE GAME MODELS
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: CREATING GAME MODELS")
    print("=" * 80)
    
    # Model 1: FAIR GAME
    # Fair probability: P(Win) = 5/6 = 0.8333, P(Lose) = 1/6 = 0.1667
    # Fair payout: 0.2x (bet ₱100, win ₱20)
    # Expected Value = (5/6 * 0.2) - (1/6 * 1) = 0.1667 - 0.1667 = 0
    fair_game = RussianRouletteGame(
        chambers=6, 
        bullets=1, 
        payout_multiplier=0.2,  # Fair payout for EV = 0
        is_fair=True
    )
    print("\n✅ Model 1: FAIR GAME")
    print(f"   - 6 chambers, 1 bullet")
    print(f"   - P(Win) = 5/6 = {fair_game.p_win:.4f}")
    print(f"   - Payout = 0.2x (win ₱20 on ₱100 bet)")
    print(f"   - Expected Value = ₱{fair_game.expected_value:.4f} per ₱1 bet")
    
    # Model 2: TWEAKED GAME - Weighted Probabilities
    # Increase bullet probability from 16.67% to 20%
    tweaked_probs_game = RussianRouletteGame(
        chambers=6,
        bullets=1,
        payout_multiplier=0.2,  # Same payout as fair game
        is_fair=False,
        weighted_probs=[0.80, 0.20]  # 80% win, 20% lose (instead of 83.33%/16.67%)
    )
    print("\n✅ Model 2: TWEAKED - Weighted Probabilities (Rigged Gun)")
    print(f"   - P(Win) reduced from 83.33% to 80%")
    print(f"   - P(Lose) increased from 16.67% to 20%")
    print(f"   - Same payout (0.2x)")
    print(f"   - Expected Value = ₱{tweaked_probs_game.expected_value:.4f} per ₱1 bet")
    print(f"   - House Edge = {tweaked_probs_game.house_edge*100:.2f}%")
    
    # ========================================================================
    # STEP 2: RUN MONTE CARLO SIMULATIONS
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: RUNNING MONTE CARLO SIMULATIONS (10,000 plays each)")
    print("=" * 80)
    
    NUM_PLAYS = 10000
    BET_AMOUNT = 100.0
    
    games = {
        '1. Fair Game': fair_game,
        '2. Weighted Probs (Rigged)': tweaked_probs_game
    }
    
    analyzer = GameAnalyzer()
    
    for name, game in games.items():
        print(f"\n🎲 Simulating {name}...")
        simulator = MonteCarloSimulator(game)
        results = simulator.run_simulation(num_plays=NUM_PLAYS, bet_per_play=BET_AMOUNT)
        stats = simulator.get_statistics()
        analyzer.add_simulation(name, game, results, stats)
        print(f"   ✓ Completed {NUM_PLAYS:,} plays")
        print(f"   ✓ House Profit: ₱{stats['house_final_balance']:,.2f}")
    
    # ========================================================================
    # STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 80)
    
    # Print comparison table
    analyzer.print_game_comparison()
    
    # Print detailed statistics
    analyzer.print_detailed_statistics()
    
    # ========================================================================
    # STEP 4: VISUALIZATIONS
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    print("\n📈 Generating cumulative balance plots...")
    analyzer.plot_cumulative_balances()
    
    print("\n📊 Generating outcome distribution plots...")
    analyzer.plot_outcome_distributions()
    
    print("\n💰 Generating house profit comparison...")
    analyzer.plot_house_profit_comparison()
    
    print("\n🔍 Generating detectability analysis (Why tweaks go unnoticed)...")
    analyzer.plot_detectability_analysis()
    
    # ========================================================================
    # STEP 5: COMPARISON & EVALUATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: COMPARISON & EVALUATION SUMMARY")
    print("=" * 80)
    
    print("\n📋 KEY FINDINGS:")
    print("-" * 60)
    
    fair_stats = analyzer.statistics['1. Fair Game']
    weighted_stats = analyzer.statistics['2. Weighted Probs (Rigged)']
    
    print(f"""
    1. FAIR GAME VALIDATION:
       - Theoretical House Edge: 0.00%
       - Empirical House Edge: {fair_stats['empirical_house_edge']*100:.2f}%
       - House Profit: ₱{fair_stats['house_final_balance']:,.2f}
       → The fair game shows near-zero house edge as expected.
    
    2. WEIGHTED PROBABILITIES (RIGGED GUN) IMPACT:
       - Theoretical House Edge: 4.00%
       - Empirical House Edge: {weighted_stats['empirical_house_edge']*100:.2f}%
       - House Profit: ₱{weighted_stats['house_final_balance']:,.2f}
       - Profit Increase vs Fair: ₱{weighted_stats['house_final_balance'] - fair_stats['house_final_balance']:,.2f}
       → Increasing bullet probability from 16.67% to 20% creates significant house edge!
       → The "rigged gun" secretly increases chance of losing by 3.33%.
    """)
    
    print("\n📊 HOUSE EDGE COMPARISON:")
    print("-" * 60)
    print(f"   {'Model':<25} {'Theoretical':>12} {'Empirical':>12}")
    print("-" * 60)
    for name in games.keys():
        game = analyzer.simulations[name]['game']
        stats = analyzer.statistics[name]
        print(f"   {name:<25} {game.house_edge*100:>11.2f}% {stats['empirical_house_edge']*100:>11.2f}%")
    
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE!")
    print("=" * 80)
    print("""
    📁 Generated Files:
       - simulation_results.png (Cumulative balance & comparison charts)
       - outcome_distributions.png (Outcome frequency distributions)
       - house_profit_comparison.png (House profit bar chart)
       - detectability_analysis.png (Why the rigged game goes unnoticed)
    
    🎯 Conclusion:
       This Monte Carlo simulation demonstrates how casinos create house edges
       through WEIGHTED PROBABILITIES (the "rigged gun" technique).
       
       By secretly increasing the bullet probability from 16.67% to 20%:
       - Players lose 3.33% more often than they expect
       - House gains a 4% edge on every bet
       - Over 10,000 plays, house profits ~₱40,000 on ₱100 bets
       
       The tweak is nearly undetectable in short sessions because
       high variance masks the small probability difference!
    """)
    
    return analyzer


# ============================================================================
# INTERACTIVE 2-PLAYER GAME (Original Feature)
# ============================================================================

class RussianRoulette:
    """Original interactive Russian Roulette game class."""
    
    def __init__(self, chambers=6, bullets=1):
        """Initialize the Russian Roulette game."""
        self.chambers = chambers
        self.bullets = bullets
        self.current_chamber = 0
        self.bullet_positions = []
        self.load_gun()
    
    def load_gun(self):
        """Randomly load bullets into the revolver chambers."""
        self.bullet_positions = random.sample(range(self.chambers), self.bullets)
        self.current_chamber = random.randint(0, self.chambers - 1)
        print(f"\n🔫 Gun loaded with {self.bullets} bullet(s) in {self.chambers} chambers...")
        print("Chamber positions are randomized!\n")
    
    def spin_cylinder(self):
        """Spin the cylinder to randomize the starting position."""
        self.current_chamber = random.randint(0, self.chambers - 1)
        print("🔄 Spinning the cylinder...")
        time.sleep(1)
    
    def pull_trigger(self):
        """Pull the trigger and check if there's a bullet."""
        result = self.current_chamber in self.bullet_positions
        self.current_chamber = (self.current_chamber + 1) % self.chambers
        return result


class TweakedRussianRoulette:
    """Tweaked Russian Roulette with configurable probabilities."""
    
    def __init__(self, chambers=6, bullets=1, even_round_prob=None, odd_round_prob=None, 
                 early_bullet_bias=0.5):
        """
        Initialize tweaked Russian Roulette game.
        
        Args:
            chambers: Number of chambers
            bullets: Number of bullets
            even_round_prob: Probability of bullet in even rounds (0-1)
            odd_round_prob: Probability of bullet in odd rounds (0-1)
            early_bullet_bias: Bias for bullet to appear early (0=late, 0.5=random, 1=early)
        """
        self.chambers = chambers
        self.bullets = bullets
        self.current_chamber = 0
        self.bullet_positions = []
        self.even_round_prob = even_round_prob
        self.odd_round_prob = odd_round_prob
        self.early_bullet_bias = early_bullet_bias
        self.chambers_checked = 0  # Track how many chambers have been checked
        self.load_gun()
    
    def load_gun(self):
        """Load bullets with bias toward early or late positions."""
        if self.early_bullet_bias == 0.5:
            # Random placement
            self.bullet_positions = random.sample(range(self.chambers), self.bullets)
        else:
            # Biased placement
            weights = []
            for i in range(self.chambers):
                if self.early_bullet_bias > 0.5:
                    # Higher bias = more likely in early chambers
                    weight = (self.chambers - i) * self.early_bullet_bias
                else:
                    # Lower bias = more likely in late chambers
                    weight = (i + 1) * (1 - self.early_bullet_bias)
                weights.append(weight)
            
            # Normalize weights
            total = sum(weights)
            probs = [w / total for w in weights]
            
            # Select bullet positions based on weights
            self.bullet_positions = np.random.choice(
                range(self.chambers), 
                size=self.bullets, 
                replace=False, 
                p=probs
            ).tolist()
        
        self.current_chamber = random.randint(0, self.chambers - 1)
        self.chambers_checked = 0  # Reset chamber counter
        print(f"\n🔫 Tweaked gun loaded with {self.bullets} bullet(s) in {self.chambers} chambers...")
        if self.early_bullet_bias > 0.6:
            print("⚠️  Bullet bias: Higher chance in EARLY rounds!")
        elif self.early_bullet_bias < 0.4:
            print("⚠️  Bullet bias: Higher chance in LATE rounds!")
        else:
            print("📊 Bullet placement: Random distribution")
        print()
    
    def spin_cylinder(self):
        """Spin the cylinder."""
        self.current_chamber = random.randint(0, self.chambers - 1)
        self.chambers_checked = 0  # Reset when spinning
        print("🔄 Spinning the cylinder...")
        time.sleep(1)
    
    def pull_trigger(self, round_number):
        """
        Pull trigger with round-dependent probability.
        
        Args:
            round_number: Current round number (1-indexed)
            
        Returns:
            bool: True if bullet fired, None if all chambers exhausted
        """
        # Check if all chambers have been exhausted
        if self.chambers_checked >= self.chambers:
            print("\n⚠️  All chambers exhausted! Automatically reloading gun...")
            time.sleep(1)
            self.load_gun()
            return None  # Signal to skip this turn
        
        # Determine if using custom probabilities for even/odd rounds
        if self.even_round_prob is not None and self.odd_round_prob is not None:
            # Use custom probabilities based on round parity
            if round_number % 2 == 0:
                # Even round
                bullet_prob = self.even_round_prob
            else:
                # Odd round
                bullet_prob = self.odd_round_prob
            
            # Weighted random outcome
            result = random.random() < bullet_prob
        else:
            # Use standard chamber-based logic
            result = self.current_chamber in self.bullet_positions
        
        self.current_chamber = (self.current_chamber + 1) % self.chambers
        self.chambers_checked += 1
        return result


class TweakedGame:
    """Tweaked 2-player game with configurable probabilities."""
    
    def __init__(self):
        """Initialize tweaked game."""
        self.players = []
        self.current_player_index = 0
        self.game_over = False
        self.round_number = 1
    
    def setup(self):
        """Set up tweaked game with custom probability configuration."""
        print("=" * 60)
        print("🎲 RUSSIAN ROULETTE - TWEAKED 2 PLAYER GAME")
        print("=" * 60)
        print("\n⚠️  WARNING: This is a simulation game only!")
        print("Never attempt this in real life.\n")
        print("This mode allows you to configure PROBABILITIES to study")
        print("how tweaking odds affects gameplay!\n")
        
        # Get player names
        player1_name = input("Enter Player 1 name: ").strip() or "Player 1"
        player2_name = input("Enter Player 2 name: ").strip() or "Player 2"
        self.players = [player1_name, player2_name]
        
        print("\n--- Basic Game Settings ---")
        chambers = self.get_valid_input("Number of chambers (3-12, default 6): ", 3, 12, 6)
        bullets = self.get_valid_input(f"Number of bullets (1-{chambers-1}, default 1): ", 1, chambers-1, 1)
        
        print("\n--- Probability Tweaking Options ---")
        print("\n[A] Configure Round-Based Probabilities (Even vs Odd rounds)")
        print("[B] Configure Bullet Position Bias (Early vs Late rounds)")
        print("[C] Use Default (Standard fair game)")
        
        tweak_choice = input("\nChoose tweaking mode (A/B/C, default C): ").strip().upper()
        
        even_round_prob = None
        odd_round_prob = None
        early_bullet_bias = 0.5
        
        if tweak_choice == 'A':
            print("\n--- Round-Based Probability Configuration ---")
            print("Set different bullet probabilities for even vs odd rounds.")
            print("Example: 30% for even rounds, 10% for odd rounds")
            print("(Fair game would be ~16.67% for both with 6 chambers, 1 bullet)\n")
            
            even_round_prob = self.get_valid_float_input(
                "Bullet probability in EVEN rounds (0-100%, default 20): ", 
                0, 100, 20
            ) / 100
            odd_round_prob = self.get_valid_float_input(
                "Bullet probability in ODD rounds (0-100%, default 10): ", 
                0, 100, 10
            ) / 100
            
            print(f"\n✅ Even rounds: {even_round_prob*100:.1f}% bullet chance")
            print(f"✅ Odd rounds: {odd_round_prob*100:.1f}% bullet chance")
        
        elif tweak_choice == 'B':
            print("\n--- Bullet Position Bias Configuration ---")
            print("Control when the bullet is more likely to appear:")
            print("  0-40%: Bullet favors LATE rounds (safer early game)")
            print("  40-60%: Random distribution (fair)")
            print("  60-100%: Bullet favors EARLY rounds (dangerous early game)\n")
            
            early_bullet_bias = self.get_valid_float_input(
                "Early bullet bias (0-100%, default 50 for random): ",
                0, 100, 50
            ) / 100
            
            if early_bullet_bias > 0.6:
                print(f"\n⚠️  Bullet bias set to {early_bullet_bias*100:.0f}% - EARLY ROUNDS ARE DANGEROUS!")
            elif early_bullet_bias < 0.4:
                print(f"\n⚠️  Bullet bias set to {early_bullet_bias*100:.0f}% - LATE ROUNDS ARE DANGEROUS!")
            else:
                print(f"\n✅ Bullet bias set to {early_bullet_bias*100:.0f}% - Random placement")
        
        else:
            print("\n✅ Using standard fair game mechanics")
        
        self.roulette = TweakedRussianRoulette(
            chambers, bullets, even_round_prob, odd_round_prob, early_bullet_bias
        )
        
        self.current_player_index = random.randint(0, 1)
        print(f"\n🎲 {self.players[self.current_player_index]} will go first!")
        time.sleep(2)
    
    def get_valid_input(self, prompt, min_val, max_val, default):
        """Get valid integer input."""
        while True:
            user_input = input(prompt).strip()
            if user_input == "":
                return default
            try:
                value = int(user_input)
                if min_val <= value <= max_val:
                    return value
                else:
                    print(f"Please enter a number between {min_val} and {max_val}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    def get_valid_float_input(self, prompt, min_val, max_val, default):
        """Get valid float input."""
        while True:
            user_input = input(prompt).strip()
            if user_input == "":
                return default
            try:
                value = float(user_input)
                if min_val <= value <= max_val:
                    return value
                else:
                    print(f"Please enter a number between {min_val} and {max_val}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    def get_current_player(self):
        """Get current player name."""
        return self.players[self.current_player_index]
    
    def switch_player(self):
        """Switch to other player."""
        self.current_player_index = (self.current_player_index + 1) % 2
    
    def play_turn(self):
        """Execute one turn with round number tracking."""
        current_player = self.get_current_player()
        other_player = self.players[(self.current_player_index + 1) % 2]
        
        print(f"\n{'=' * 60}")
        print(f"🎯 {current_player}'s turn (Round {self.round_number})")
        if self.round_number % 2 == 0:
            print("📊 EVEN ROUND", end="")
        else:
            print("📊 ODD ROUND", end="")
        
        # Show probability if configured
        if self.roulette.even_round_prob is not None:
            prob = self.roulette.even_round_prob if self.round_number % 2 == 0 else self.roulette.odd_round_prob
            print(f" - Bullet probability: {prob*100:.1f}%")
        else:
            print()
        print(f"{'=' * 60}")
        
        spin_choice = input("\nDo you want to spin the cylinder? (y/n, default n): ").strip().lower()
        if spin_choice == 'y':
            self.roulette.spin_cylinder()
        
        input(f"\n{current_player}, press Enter to pull the trigger...")
        
        print("\n💀 Pulling trigger...")
        time.sleep(1)
        
        result = self.roulette.pull_trigger(self.round_number)
        
        # Handle gun reload scenario
        if result is None:
            print("🔄 Gun reloaded! Turn skipped, continuing game...")
            time.sleep(1)
            return  # Don't switch players, just continue
        
        if result:
            print("\n💥 BANG! 💥")
            time.sleep(1)
            print(f"\n☠️  {current_player} has been eliminated in Round {self.round_number}!")
            print(f"🏆 {other_player} WINS! 🏆")
            self.game_over = True
        else:
            print("\n🔘 *CLICK* - Empty chamber!")
            print(f"✅ {current_player} survives Round {self.round_number}!")
            time.sleep(1)
            self.switch_player()
    
    def play(self):
        """Main game loop."""
        self.setup()
        
        self.round_number = 1
        while not self.game_over:
            print(f"\n\n{'#' * 60}")
            print(f"ROUND {self.round_number}")
            print(f"{'#' * 60}")
            self.play_turn()
            self.round_number += 1
        
        print("\n" + "=" * 60)
        print("GAME OVER")
        print("=" * 60)
        
        play_again = input("\nDo you want to play again? (y/n): ").strip().lower()
        if play_again == 'y':
            self.game_over = False
            self.round_number = 1
            self.play()


class Game:
    """Interactive 2-player game class."""
    
    def __init__(self):
        """Initialize the game with 2 players."""
        self.players = []
        self.current_player_index = 0
        self.game_over = False
    
    def setup(self):
        """Set up the game with player names and game settings."""
        print("=" * 50)
        print("🎮 RUSSIAN ROULETTE - 2 PLAYER GAME")
        print("=" * 50)
        print("\nWARNING: This is a simulation game only!")
        print("Never attempt this in real life.\n")
        
        player1_name = input("Enter Player 1 name: ").strip() or "Player 1"
        player2_name = input("Enter Player 2 name: ").strip() or "Player 2"
        self.players = [player1_name, player2_name]
        
        print("\n--- Game Settings ---")
        chambers = self.get_valid_input("Number of chambers (3-12, default 6): ", 3, 12, 6)
        bullets = self.get_valid_input(f"Number of bullets (1-{chambers-1}, default 1): ", 1, chambers-1, 1)
        
        self.roulette = RussianRoulette(chambers, bullets)
        
        self.current_player_index = random.randint(0, 1)
        print(f"\n🎲 {self.players[self.current_player_index]} will go first!")
        time.sleep(2)
    
    def get_valid_input(self, prompt, min_val, max_val, default):
        """Get valid integer input from user."""
        while True:
            user_input = input(prompt).strip()
            if user_input == "":
                return default
            try:
                value = int(user_input)
                if min_val <= value <= max_val:
                    return value
                else:
                    print(f"Please enter a number between {min_val} and {max_val}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    def get_current_player(self):
        """Get the name of the current player."""
        return self.players[self.current_player_index]
    
    def switch_player(self):
        """Switch to the other player."""
        self.current_player_index = (self.current_player_index + 1) % 2
    
    def play_turn(self):
        """Execute one turn of the game."""
        current_player = self.get_current_player()
        other_player = self.players[(self.current_player_index + 1) % 2]
        
        print(f"\n{'=' * 50}")
        print(f"🎯 {current_player}'s turn")
        print(f"{'=' * 50}")
        
        spin_choice = input("\nDo you want to spin the cylinder? (y/n, default n): ").strip().lower()
        if spin_choice == 'y':
            self.roulette.spin_cylinder()
        
        input(f"\n{current_player}, press Enter to pull the trigger...")
        
        print("\n💀 Pulling trigger...")
        time.sleep(1)
        
        if self.roulette.pull_trigger():
            print("\n💥 BANG! 💥")
            time.sleep(1)
            print(f"\n☠️  {current_player} has been eliminated!")
            print(f"🏆 {other_player} WINS! 🏆")
            self.game_over = True
        else:
            print("\n🔘 *CLICK* - Empty chamber!")
            print(f"✅ {current_player} survives this round.")
            time.sleep(1)
            self.switch_player()
    
    def play(self):
        """Main game loop."""
        self.setup()
        
        round_number = 1
        while not self.game_over:
            print(f"\n\n{'#' * 50}")
            print(f"ROUND {round_number}")
            print(f"{'#' * 50}")
            self.play_turn()
            round_number += 1
        
        print("\n" + "=" * 50)
        print("GAME OVER")
        print("=" * 50)
        
        play_again = input("\nDo you want to play again? (y/n): ").strip().lower()
        if play_again == 'y':
            self.game_over = False
            self.play()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point - choose between simulation or interactive game."""
    
    print("\n" + "=" * 70)
    print("RUSSIAN ROULETTE - CSEC 413 FINAL PROJECT")
    print("=" * 70)
    print("""
    Choose an option:
    
    [1] Run Monte Carlo Simulation & Analysis
        (10,000+ plays, EDA, visualizations with fair vs tweaked)
    
    [2] Play Interactive 2-Player Game (Standard)
        (Original fair game mode)
    
    [3] Play Tweaked 2-Player Game (Configurable Probabilities)
        (Customize even/odd round probabilities & bullet bias)
    
    [4] Exit
    """)
    
    choice = input("Enter your choice (1/2/3/4): ").strip()
    
    if choice == '1':
        run_complete_analysis()
    elif choice == '2':
        game = Game()
        game.play()
        print("\nThanks for playing! Remember: This is just a game - never try this in real life!")
    elif choice == '3':
        tweaked_game = TweakedGame()
        tweaked_game.play()
        print("\nThanks for playing! Remember: This is just a game - never try this in real life!")
    elif choice == '4':
        print("\nGoodbye!")
    else:
        print("\nInvalid choice. Running simulation by default...")
        run_complete_analysis()


if __name__ == "__main__":
    main()
