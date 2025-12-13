"""
Russian Roulette Game - Flask Web Application
CSEC 413 - Modeling and Simulation Final Project

Web-based version with:
- Monte Carlo Simulation & EDA (Mode 1)
- Interactive 2-Player Game (Mode 2)
- Tweaked Game with configurable probabilities (Mode 3)
- Casino Mode: Bet against Computer (Mode 4)
"""

from flask import Flask, render_template, request, session, jsonify, redirect, url_for
import random
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'russian_roulette_secret_key_2025'


# ============================================================================
# MONTE CARLO SIMULATION CLASSES
# ============================================================================

class RussianRouletteGame:
    """
    Base class for Russian Roulette Casino Game simulation.
    """
    
    def __init__(self, chambers=6, bullets=1, payout_multiplier=0.2, 
                 is_fair=True, weighted_probs=None):
        self.chambers = chambers
        self.bullets = bullets
        self.payout_multiplier = payout_multiplier
        self.is_fair = is_fair
        
        if weighted_probs is not None:
            self.p_win = weighted_probs[0]
            self.p_lose = weighted_probs[1]
        else:
            self.p_win = (chambers - bullets) / chambers
            self.p_lose = bullets / chambers
        
        self.expected_value = self.p_win * self.payout_multiplier - self.p_lose
        self.house_edge = -self.expected_value
    
    def play_single_round(self, bet=1.0):
        outcome = np.random.choice(['win', 'lose'], p=[self.p_win, self.p_lose])
        if outcome == 'win':
            return True, bet * self.payout_multiplier
        else:
            return False, -bet
    
    def get_game_info(self):
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


class MonteCarloSimulator:
    """Monte Carlo Simulation Engine for Russian Roulette games."""
    
    def __init__(self, game):
        self.game = game
        self.results = None
    
    def run_simulation(self, num_plays=10000, bet_per_play=100.0):
        results = []
        cumulative_player_balance = 0
        cumulative_house_balance = 0
        
        for play_num in range(1, num_plays + 1):
            won, net_result = self.game.play_single_round(bet_per_play)
            cumulative_player_balance += net_result
            cumulative_house_balance -= net_result
            
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
    
    def get_statistics(self):
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


def run_monte_carlo_simulation(num_plays=10000, bet_amount=100):
    """Run complete Monte Carlo simulation and return results."""
    
    # Model 1: Fair Game
    fair_game = RussianRouletteGame(
        chambers=6, bullets=1, payout_multiplier=0.2, is_fair=True
    )
    
    # Model 2: Weighted Probabilities (Rigged)
    tweaked_game = RussianRouletteGame(
        chambers=6, bullets=1, payout_multiplier=0.2,
        is_fair=False, weighted_probs=[0.80, 0.20]
    )
    
    # Run simulations
    fair_simulator = MonteCarloSimulator(fair_game)
    fair_results = fair_simulator.run_simulation(num_plays, bet_amount)
    fair_stats = fair_simulator.get_statistics()
    
    tweaked_simulator = MonteCarloSimulator(tweaked_game)
    tweaked_results = tweaked_simulator.run_simulation(num_plays, bet_amount)
    tweaked_stats = tweaked_simulator.get_statistics()
    
    return {
        'fair': {
            'game': fair_game,
            'results': fair_results,
            'stats': fair_stats
        },
        'tweaked': {
            'game': tweaked_game,
            'results': tweaked_results,
            'stats': tweaked_stats
        }
    }


def generate_simulation_plots(simulation_data):
    """Generate all simulation plots and return as base64 encoded images."""
    plots = {}
    
    fair_results = simulation_data['fair']['results']
    tweaked_results = simulation_data['tweaked']['results']
    fair_stats = simulation_data['fair']['stats']
    tweaked_stats = simulation_data['tweaked']['stats']
    fair_game = simulation_data['fair']['game']
    tweaked_game = simulation_data['tweaked']['game']
    
    # Plot 1: Cumulative Balances
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Russian Roulette Monte Carlo Simulation Results', fontsize=14, fontweight='bold')
    
    # Player balance over time
    ax1 = axes[0, 0]
    ax1.plot(fair_results['play_number'], fair_results['cumulative_player_balance'], 
             label='Fair Game', color='green', alpha=0.8)
    ax1.plot(tweaked_results['play_number'], tweaked_results['cumulative_player_balance'], 
             label='Weighted Probs (Rigged)', color='red', alpha=0.8)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Number of Plays')
    ax1.set_ylabel('Player Cumulative Balance (₱)')
    ax1.set_title('Player Balance Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # House balance over time
    ax2 = axes[0, 1]
    ax2.plot(fair_results['play_number'], fair_results['cumulative_house_balance'], 
             label='Fair Game', color='green', alpha=0.8)
    ax2.plot(tweaked_results['play_number'], tweaked_results['cumulative_house_balance'], 
             label='Weighted Probs (Rigged)', color='red', alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Number of Plays')
    ax2.set_ylabel('House Cumulative Balance (₱)')
    ax2.set_title('House Profit Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Win rate comparison
    ax3 = axes[1, 0]
    names = ['Fair Game', 'Weighted Probs']
    win_rates = [fair_stats['win_rate'] * 100, tweaked_stats['win_rate'] * 100]
    theoretical_rates = [fair_game.p_win * 100, tweaked_game.p_win * 100]
    x = np.arange(len(names))
    width = 0.35
    ax3.bar(x - width/2, win_rates, width, label='Empirical', color='steelblue')
    ax3.bar(x + width/2, theoretical_rates, width, label='Theoretical', color='lightcoral')
    ax3.set_ylabel('Win Rate (%)')
    ax3.set_title('Win Rate: Empirical vs Theoretical')
    ax3.set_xticks(x)
    ax3.set_xticklabels(names)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # House edge comparison
    ax4 = axes[1, 1]
    house_edges = [fair_stats['empirical_house_edge'] * 100, tweaked_stats['empirical_house_edge'] * 100]
    theoretical_edges = [fair_game.house_edge * 100, tweaked_game.house_edge * 100]
    ax4.bar(x - width/2, house_edges, width, label='Empirical', color='darkgreen')
    ax4.bar(x + width/2, theoretical_edges, width, label='Theoretical', color='lightgreen')
    ax4.set_ylabel('House Edge (%)')
    ax4.set_title('House Edge: Empirical vs Theoretical')
    ax4.set_xticks(x)
    ax4.set_xticklabels(names)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plots['cumulative'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    # Plot 2: Outcome Distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Distribution of Single Play Outcomes', fontsize=14, fontweight='bold')
    
    axes[0].hist(fair_results['net_result'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(x=fair_results['net_result'].mean(), color='red', linestyle='--', 
                   label=f'Mean: ₱{fair_results["net_result"].mean():.2f}')
    axes[0].set_xlabel('Net Result (₱)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Fair Game\nMean: ₱{fair_results["net_result"].mean():.2f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(tweaked_results['net_result'], bins=30, edgecolor='black', alpha=0.7, color='coral')
    axes[1].axvline(x=tweaked_results['net_result'].mean(), color='red', linestyle='--',
                   label=f'Mean: ₱{tweaked_results["net_result"].mean():.2f}')
    axes[1].set_xlabel('Net Result (₱)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Weighted Probs (Rigged)\nMean: ₱{tweaked_results["net_result"].mean():.2f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plots['distributions'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    # Plot 3: House Profit Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    names = ['Fair Game', 'Weighted Probs (Rigged)']
    profits = [fair_stats['house_final_balance'], tweaked_stats['house_final_balance']]
    colors = ['green' if p > 0 else 'red' for p in profits]
    bars = ax.bar(names, profits, color=['green', 'red'], edgecolor='black', alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('House Final Balance (₱)')
    ax.set_title(f'Final House Profit Comparison (After {fair_stats["total_plays"]:,} Plays)\n₱100 Bet Per Play', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, profit in zip(bars, profits):
        height = bar.get_height()
        ax.annotate(f'₱{profit:,.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 5 if height >= 0 else -15), textcoords="offset points",
                   ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plots['profit_comparison'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    # Plot 4: Detectability Analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Why Players Don\'t Notice the Rigged Gun', fontsize=14, fontweight='bold')
    
    # Convergence over time
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
    
    # Distribution overlap
    ax2 = axes[0, 1]
    ax2.hist(fair_results['net_result'], bins=30, alpha=0.6, label='Fair Game', color='green', edgecolor='black')
    ax2.hist(tweaked_results['net_result'], bins=30, alpha=0.6, label='Rigged Gun', color='red', edgecolor='black')
    ax2.set_xlabel('Single Play Result (₱)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Outcome Distribution Looks Almost Identical!')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Short session simulation
    ax3 = axes[1, 0]
    session_length = 20
    num_sessions = 100
    fair_session_results = []
    tweaked_session_results = []
    for i in range(num_sessions):
        start_idx = i * session_length % (len(fair_results) - session_length)
        fair_session_results.append(fair_results['net_result'][start_idx:start_idx+session_length].sum())
        tweaked_session_results.append(tweaked_results['net_result'][start_idx:start_idx+session_length].sum())
    
    ax3.hist(fair_session_results, bins=20, alpha=0.6, label='Fair Game', color='green', edgecolor='black')
    ax3.hist(tweaked_session_results, bins=20, alpha=0.6, label='Rigged Gun', color='red', edgecolor='black')
    ax3.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax3.set_xlabel(f'Total Profit/Loss in {session_length}-Play Session (₱)')
    ax3.set_ylabel('Number of Sessions')
    ax3.set_title(f'Short Sessions ({session_length} plays): Results Vary Wildly!')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Long-term truth
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
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plots['detectability'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return plots


# ============================================================================
# GAME STATE CLASS FOR INTERACTIVE GAME
# ============================================================================

class GameState:
    """Game state for interactive 2-player game."""
    
    def __init__(self, player1_name, player2_name, chambers=6, bullets=1, 
                 even_round_prob=None, odd_round_prob=None, early_bullet_bias=0.5,
                 is_casino_mode=False, starting_balance=1000, initial_bet=100):
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.players = [player1_name, player2_name]
        self.current_player_index = 0  # Boy (Player 1) always goes first
        self.chambers = chambers
        self.bullets = bullets
        self.even_round_prob = even_round_prob
        self.odd_round_prob = odd_round_prob
        self.early_bullet_bias = early_bullet_bias
        self.round_number = 1
        self.chambers_checked = 0
        self.game_over = False
        self.winner = None
        self.loser = None
        self.bullet_positions = []
        self.current_chamber = 0
        self.image_phase = 'start'
        self.mode = '2'  # Default to standard game
        
        # Casino mode fields
        self.is_casino_mode = is_casino_mode
        self.starting_balance = starting_balance
        self.current_balance = starting_balance
        self.current_bet = initial_bet
        self.initial_bet = initial_bet
        self.total_wins = 0
        self.total_losses = 0
        
        self.load_gun()
        
    def load_gun(self):
        """Load bullets with optional bias."""
        if self.early_bullet_bias == 0.5:
            self.bullet_positions = random.sample(range(self.chambers), self.bullets)
        else:
            weights = []
            for i in range(self.chambers):
                if self.early_bullet_bias > 0.5:
                    weight = (self.chambers - i) * self.early_bullet_bias
                else:
                    weight = (i + 1) * (1 - self.early_bullet_bias)
                weights.append(weight)
            
            total = sum(weights)
            probs = [w / total for w in weights]
            
            # Use weighted selection
            self.bullet_positions = np.random.choice(
                range(self.chambers), size=self.bullets, replace=False, p=probs
            ).tolist()
        
        self.current_chamber = random.randint(0, self.chambers - 1)
        self.chambers_checked = 0
    
    def pull_trigger(self):
        """Pull trigger and return result."""
        if self.chambers_checked >= self.chambers:
            self.load_gun()
            return None  # Reload signal
        
        # Check for custom probabilities (Mode 3)
        if self.even_round_prob is not None and self.odd_round_prob is not None:
            if self.round_number % 2 == 0:
                bullet_prob = self.even_round_prob
            else:
                bullet_prob = self.odd_round_prob
            result = random.random() < bullet_prob
        else:
            result = self.current_chamber in self.bullet_positions
        
        self.current_chamber = (self.current_chamber + 1) % self.chambers
        self.chambers_checked += 1
        return result
    
    def get_current_player(self):
        return self.players[self.current_player_index]
    
    def get_other_player(self):
        return self.players[(self.current_player_index + 1) % 2]
    
    def switch_player(self):
        self.current_player_index = (self.current_player_index + 1) % 2
    
    def to_dict(self):
        return {
            'player1_name': self.player1_name,
            'player2_name': self.player2_name,
            'current_player_index': self.current_player_index,
            'chambers': self.chambers,
            'bullets': self.bullets,
            'even_round_prob': self.even_round_prob,
            'odd_round_prob': self.odd_round_prob,
            'early_bullet_bias': self.early_bullet_bias,
            'round_number': self.round_number,
            'chambers_checked': self.chambers_checked,
            'game_over': self.game_over,
            'winner': self.winner,
            'loser': self.loser,
            'bullet_positions': self.bullet_positions,
            'current_chamber': self.current_chamber,
            'image_phase': self.image_phase,
            'mode': self.mode,
            'is_casino_mode': self.is_casino_mode,
            'starting_balance': self.starting_balance,
            'current_balance': self.current_balance,
            'current_bet': self.current_bet,
            'initial_bet': self.initial_bet,
            'total_wins': self.total_wins,
            'total_losses': self.total_losses,
        }
    
    @staticmethod
    def from_dict(data):
        game = GameState(
            data['player1_name'],
            data['player2_name'],
            data['chambers'],
            data['bullets'],
            data['even_round_prob'],
            data['odd_round_prob'],
            data['early_bullet_bias'],
            data.get('is_casino_mode', False),
            data.get('starting_balance', 1000),
            data.get('initial_bet', 100)
        )
        game.current_player_index = data['current_player_index']
        game.round_number = data['round_number']
        game.chambers_checked = data['chambers_checked']
        game.game_over = data['game_over']
        game.winner = data['winner']
        game.loser = data.get('loser')
        game.bullet_positions = data['bullet_positions']
        game.current_chamber = data['current_chamber']
        game.image_phase = data.get('image_phase', 'getting_gun')
        game.mode = data.get('mode', '2')
        game.is_casino_mode = data.get('is_casino_mode', False)
        game.current_balance = data.get('current_balance', data.get('starting_balance', 1000))
        game.current_bet = data.get('current_bet', data.get('initial_bet', 100))
        game.total_wins = data.get('total_wins', 0)
        game.total_losses = data.get('total_losses', 0)
        return game


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_game_state():
    if 'game_state' not in session:
        return None
    return GameState.from_dict(session['game_state'])


def save_game_state(game_state):
    session['game_state'] = game_state.to_dict()
    session.modified = True


def get_image_sequence(game_state):
    """Get appropriate image based on game phase."""
    current_player_is_boy = (game_state.current_player_index == 0)
    
    if game_state.round_number == 1 and game_state.image_phase == 'start':
        return "first_image_gun_in_table.png"
    
    if game_state.image_phase == 'getting_gun':
        if current_player_is_boy:
            return "2nd_boy_getting_gun.png"
        else:
            return "2nd_girl_getting_gun.png"
    elif game_state.image_phase == 'pointing':
        if current_player_is_boy:
            return "3rd_gun_pointed_to_boy.png"
        else:
            return "3rd_gun_pointed_to_girl.png"
    
    return "first_image_gun_in_table.png"


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Home page - mode selection"""
    return render_template('index.html')


@app.route('/setup', methods=['GET', 'POST'])
def setup():
    """Game setup page"""
    if request.method == 'POST':
        data = request.json
        mode = data.get('mode', '2')
        player1_name = data.get('player1_name', 'Boy')
        player2_name = data.get('player2_name', 'Girl')
        chambers = int(data.get('chambers', 6))
        bullets = int(data.get('bullets', 1))
        
        even_round_prob = None
        odd_round_prob = None
        early_bullet_bias = 0.5
        is_casino_mode = False
        starting_balance = 1000
        initial_bet = 100
        
        if mode == '3' or mode == '4':
            config_mode = data.get('config_mode', 'C')
            
            if config_mode == 'A':
                even_round_prob = float(data.get('even_round_prob', 20)) / 100
                odd_round_prob = float(data.get('odd_round_prob', 10)) / 100
            elif config_mode == 'B':
                early_bullet_bias = float(data.get('early_bullet_bias', 50)) / 100
        
        if mode == '4':
            is_casino_mode = True
            starting_balance = int(data.get('starting_balance', 1000))
            initial_bet = int(data.get('initial_bet', 100))
            player2_name = 'Computer 🤖'
        
        game = GameState(
            player1_name, player2_name, chambers, bullets,
            even_round_prob, odd_round_prob, early_bullet_bias,
            is_casino_mode, starting_balance, initial_bet
        )
        game.mode = mode
        
        save_game_state(game)
        return jsonify({'status': 'success', 'redirect': url_for('game')})
    
    return render_template('setup.html')


@app.route('/game')
def game():
    """Main game page"""
    game_state = get_game_state()
    if not game_state:
        return redirect(url_for('index'))
    
    image_sequence = get_image_sequence(game_state)
    current_player_is_boy = (game_state.current_player_index == 0)
    
    # Calculate probability info for display
    prob_info = None
    if game_state.even_round_prob is not None:
        if game_state.round_number % 2 == 0:
            prob_info = f"EVEN Round - {game_state.even_round_prob * 100:.1f}% bullet chance"
        else:
            prob_info = f"ODD Round - {game_state.odd_round_prob * 100:.1f}% bullet chance"
    elif game_state.early_bullet_bias != 0.5:
        if game_state.early_bullet_bias > 0.6:
            prob_info = f"Bullet bias: {game_state.early_bullet_bias * 100:.0f}% - EARLY rounds are dangerous!"
        elif game_state.early_bullet_bias < 0.4:
            prob_info = f"Bullet bias: {game_state.early_bullet_bias * 100:.0f}% - LATE rounds are dangerous!"
    
    game_info = {
        'round': game_state.round_number,
        'current_player': game_state.get_current_player(),
        'current_player_is_boy': current_player_is_boy,
        'player1_name': game_state.player1_name,
        'player2_name': game_state.player2_name,
        'chambers': game_state.chambers,
        'bullets': game_state.bullets,
        'chambers_checked': game_state.chambers_checked,
        'image': image_sequence,
        'image_phase': game_state.image_phase,
        'game_over': game_state.game_over,
        'winner': game_state.winner,
        'mode': game_state.mode,
        'prob_info': prob_info,
        # Casino mode info
        'is_casino_mode': game_state.is_casino_mode,
        'current_balance': game_state.current_balance,
        'current_bet': game_state.current_bet,
        'total_wins': game_state.total_wins,
        'total_losses': game_state.total_losses,
    }
    
    return render_template('game.html', game_info=game_info)


@app.route('/api/pull_trigger', methods=['POST'])
def pull_trigger():
    """Handle trigger pull"""
    game_state = get_game_state()
    if not game_state or game_state.game_over:
        return jsonify({'status': 'error', 'message': 'Game not in progress'})
    
    current_player = game_state.get_current_player()
    other_player = game_state.get_other_player()
    current_player_is_boy = (game_state.current_player_index == 0)
    
    result = game_state.pull_trigger()
    
    if result is None:
        game_state.round_number += 1
        game_state.image_phase = 'getting_gun'
        save_game_state(game_state)
        return jsonify({
            'status': 'reload',
            'message': '⚠️ All chambers exhausted! Gun reloaded.',
            'next_round': game_state.round_number,
            'image': get_image_sequence(game_state)
        })
    
    if result:
        game_state.game_over = True
        game_state.winner = other_player
        game_state.loser = current_player
        game_state.image_phase = 'outcome'
        image = f"4th_gun_shot_{'boy' if current_player_is_boy else 'girl'}.png"
        message = f"💥 BANG! {current_player} has been eliminated!"
        
        # Casino mode: handle win/loss
        if game_state.is_casino_mode:
            if current_player_is_boy:  # Player 1 (human) lost
                game_state.current_balance -= game_state.current_bet
                game_state.total_losses += 1
                message += f" You lost ₱{game_state.current_bet:,}!"
            else:  # Computer lost, player wins
                game_state.current_balance += game_state.current_bet
                game_state.total_wins += 1
                message += f" You won ₱{game_state.current_bet:,}!"
    else:
        if current_player_is_boy:
            image = "4th_gun_shot_safe_bring_back_gun_to_table.boy.png"
        else:
            image = "4th_gun_shot_safe_bring_back_gun_to_table_girl.png"
        message = f"✅ {current_player} survives Round {game_state.round_number}!"
        game_state.switch_player()
        game_state.round_number += 1
        game_state.image_phase = 'getting_gun'
    
    save_game_state(game_state)
    
    return jsonify({
        'status': 'success',
        'result': 'bang' if result else 'safe',
        'message': message,
        'image': image,
        'current_player': game_state.get_current_player(),
        'current_player_is_boy': (game_state.current_player_index == 0),
        'game_over': game_state.game_over,
        'winner': game_state.winner,
        'round': game_state.round_number,
        # Casino mode info
        'is_casino_mode': game_state.is_casino_mode,
        'current_balance': game_state.current_balance,
        'current_bet': game_state.current_bet,
        'total_wins': game_state.total_wins,
        'total_losses': game_state.total_losses,
    })


@app.route('/api/advance_phase', methods=['POST'])
def advance_phase():
    """Advance image phase"""
    game_state = get_game_state()
    if not game_state or game_state.game_over:
        return jsonify({'status': 'error', 'message': 'Game not in progress'})
    
    if game_state.image_phase == 'start':
        game_state.image_phase = 'getting_gun'
    elif game_state.image_phase == 'getting_gun':
        game_state.image_phase = 'pointing'
    
    save_game_state(game_state)
    
    return jsonify({
        'status': 'success',
        'image': get_image_sequence(game_state),
        'phase': game_state.image_phase
    })


@app.route('/game_over')
def game_over():
    """Game over page"""
    game_state = get_game_state()
    if not game_state or not game_state.game_over:
        return redirect(url_for('game'))
    
    return render_template('game_over.html', 
                         winner=game_state.winner,
                         loser=game_state.loser)


@app.route('/new_game')
def new_game():
    """Start a new game"""
    session.clear()
    return redirect(url_for('index'))


# ============================================================================
# CASINO MODE ROUTES
# ============================================================================

@app.route('/api/casino_rematch', methods=['POST'])
def casino_rematch():
    """Start a new round in casino mode with same settings"""
    game_state = get_game_state()
    if not game_state or not game_state.is_casino_mode:
        return jsonify({'status': 'error', 'message': 'Not in casino mode'})
    
    # Check if player has enough balance
    if game_state.current_balance < game_state.current_bet:
        return jsonify({'status': 'error', 'message': 'Insufficient balance!'})
    
    # Reset game state for new round
    game_state.game_over = False
    game_state.winner = None
    game_state.loser = None
    game_state.round_number = 1
    game_state.current_player_index = 0  # Player always goes first
    game_state.image_phase = 'start'
    game_state.load_gun()
    
    save_game_state(game_state)
    
    return jsonify({
        'status': 'success',
        'message': 'New game started!',
        'redirect': url_for('game')
    })


@app.route('/api/change_bet', methods=['POST'])
def change_bet():
    """Change bet amount in casino mode"""
    game_state = get_game_state()
    if not game_state or not game_state.is_casino_mode:
        return jsonify({'status': 'error', 'message': 'Not in casino mode'})
    
    data = request.json
    new_bet = int(data.get('new_bet', game_state.current_bet))
    
    # Validate bet
    if new_bet < 10:
        return jsonify({'status': 'error', 'message': 'Minimum bet is ₱10'})
    if new_bet > game_state.current_balance:
        return jsonify({'status': 'error', 'message': 'Bet cannot exceed balance!'})
    
    game_state.current_bet = new_bet
    save_game_state(game_state)
    
    return jsonify({
        'status': 'success',
        'new_bet': new_bet,
        'message': f'Bet changed to ₱{new_bet:,}'
    })


# ============================================================================
# SIMULATION ROUTES
# ============================================================================

@app.route('/simulation')
def simulation():
    """Monte Carlo simulation page"""
    return render_template('simulation.html')


@app.route('/api/run_simulation', methods=['POST'])
def api_run_simulation():
    """Run Monte Carlo simulation via API"""
    data = request.json
    num_plays = int(data.get('num_plays', 10000))
    bet_amount = float(data.get('bet_amount', 100))
    
    # Run simulation
    simulation_data = run_monte_carlo_simulation(num_plays, bet_amount)
    
    # Generate plots
    plots = generate_simulation_plots(simulation_data)
    
    # Prepare statistics for display
    fair_stats = simulation_data['fair']['stats']
    tweaked_stats = simulation_data['tweaked']['stats']
    fair_game = simulation_data['fair']['game']
    tweaked_game = simulation_data['tweaked']['game']
    
    return jsonify({
        'status': 'success',
        'plots': plots,
        'fair_stats': {
            'total_plays': fair_stats['total_plays'],
            'wins': int(fair_stats['wins']),
            'losses': int(fair_stats['losses']),
            'win_rate': round(fair_stats['win_rate'] * 100, 2),
            'house_final_balance': round(fair_stats['house_final_balance'], 2),
            'player_final_balance': round(fair_stats['player_final_balance'], 2),
            'empirical_house_edge': round(fair_stats['empirical_house_edge'] * 100, 2),
            'theoretical_house_edge': round(fair_game.house_edge * 100, 2),
            'p_win': round(fair_game.p_win * 100, 2),
            'p_lose': round(fair_game.p_lose * 100, 2),
        },
        'tweaked_stats': {
            'total_plays': tweaked_stats['total_plays'],
            'wins': int(tweaked_stats['wins']),
            'losses': int(tweaked_stats['losses']),
            'win_rate': round(tweaked_stats['win_rate'] * 100, 2),
            'house_final_balance': round(tweaked_stats['house_final_balance'], 2),
            'player_final_balance': round(tweaked_stats['player_final_balance'], 2),
            'empirical_house_edge': round(tweaked_stats['empirical_house_edge'] * 100, 2),
            'theoretical_house_edge': round(tweaked_game.house_edge * 100, 2),
            'p_win': round(tweaked_game.p_win * 100, 2),
            'p_lose': round(tweaked_game.p_lose * 100, 2),
        }
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
