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
    Russian Roulette Game for Monte Carlo simulation.
    
    Models:
    - Fair Game: 50/50 distribution (bullet equally likely in any chamber)
    - Tweaked Game: Even/Odd weighted (e.g., 65% even rounds, 35% odd rounds)
    
    In tweaked mode, Player goes first (Round 1 = odd), Computer goes second (Round 2 = even).
    
    The even_prob represents the probability that the bullet is in an EVEN chamber.
    The odd_prob is automatically (1 - even_prob) since total must equal 100%.
    
    If bullet is more likely in ODD rounds (player's turns), player has higher chance of losing.
    So higher odd_prob = house advantage.
    """
    
    def __init__(self, chambers=6, bullets=1, is_fair=True, even_prob=0.5):
        self.chambers = chambers
        self.bullets = bullets
        self.is_fair = is_fair
        
        # Ensure probabilities sum to 1.0
        self.even_prob = even_prob  # Probability bullet is in even chambers (2,4,6)
        self.odd_prob = 1.0 - even_prob  # Probability bullet is in odd chambers (1,3,5)
        
        # In a fair 6-chamber game with 1 bullet:
        # - 3 odd chambers (1,3,5) = Player's turns
        # - 3 even chambers (2,4,6) = Computer's turns
        # 
        # Player wins if bullet is in an even chamber (computer gets shot)
        # Player loses if bullet is in an odd chamber (player gets shot)
        
        if is_fair:
            # Fair game: 50% chance for each side
            self.p_win = 0.5  # Player wins 50%
            self.p_lose = 0.5  # Player loses 50%
        else:
            # Tweaked: bullet placement weighted by even/odd
            # Player wins when bullet is in EVEN chamber (computer's turn)
            # Player loses when bullet is in ODD chamber (player's turn)
            self.p_win = self.even_prob   # Player wins if bullet in even (computer's turn)
            self.p_lose = self.odd_prob   # Player loses if bullet in odd (player's turn)
        
    def simulate_full_game(self):
        """
        Simulate a complete game between Player and Computer.
        
        The bullet is placed according to even/odd probability distribution.
        Then players take turns until someone gets shot.
        
        Returns: (player_won, rounds_played, who_got_shot)
        """
        # First, determine where the bullet is based on probability distribution
        if self.is_fair:
            # Fair: bullet equally likely in any chamber
            bullet_in_even = random.random() < 0.5
        else:
            # Tweaked: bullet placement weighted
            bullet_in_even = random.random() < self.even_prob
        
        # Now simulate the game
        # If bullet is in even chamber, computer (who plays even rounds) gets shot
        # If bullet is in odd chamber, player (who plays odd rounds) gets shot
        
        if bullet_in_even:
            # Bullet is in an even chamber - determine which one (2, 4, or 6)
            chamber = random.choice([2, 4, 6])
            return True, chamber, 'computer'  # Player wins, computer got shot
        else:
            # Bullet is in an odd chamber - determine which one (1, 3, or 5)
            chamber = random.choice([1, 3, 5])
            return False, chamber, 'player'  # Player loses, player got shot
    
    def play_single_round(self, bet=1.0):
        """Play one complete game and return result."""
        player_won, rounds, who_shot = self.simulate_full_game()
        if player_won:
            return True, bet  # Player wins their bet back
        else:
            return False, -bet  # Player loses their bet
    
    def get_game_info(self):
        return {
            'chambers': self.chambers,
            'bullets': self.bullets,
            'is_fair': self.is_fair,
            'even_prob': self.even_prob,
            'odd_prob': self.odd_prob,
            'p_win': self.p_win,
            'p_lose': self.p_lose,
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
            'empirical_win_rate': wins / total_plays,
            'theoretical_win_rate': self.game.p_win,
        }


def run_monte_carlo_simulation(num_plays=10000, bet_amount=100, even_prob=0.35):
    """
    Run complete Monte Carlo simulation comparing Fair vs Tweaked games.
    
    Fair Game: 50/50 distribution - bullet equally likely in even or odd chambers
    Tweaked Game: Weighted by even/odd chambers
    
    Parameters:
        even_prob: Probability bullet is in an EVEN chamber (computer's turn)
                   odd_prob is automatically (1 - even_prob)
    
    Example with even_prob=0.35:
    - 35% chance bullet is in EVEN chambers (2,4,6) = Computer gets shot
    - 65% chance bullet is in ODD chambers (1,3,5) = Player gets shot
    - This means Player wins 35% of the time, loses 65%
    - This setup FAVORS the HOUSE (computer)!
    
    For house advantage: use even_prob < 0.5 (bullet more likely in player's chambers)
    """
    
    # Model 1: Fair Game (50/50 distribution)
    fair_game = RussianRouletteGame(
        chambers=6, bullets=1, is_fair=True
    )
    
    # Model 2: Tweaked Game - House Advantage
    # Lower even_prob means Player is more likely to get shot (bullet in odd chambers)
    tweaked_game = RussianRouletteGame(
        chambers=6, bullets=1, is_fair=False,
        even_prob=even_prob  # odd_prob is automatically 1 - even_prob
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
    names = ['Fair Game', 'Tweaked (Even/Odd)']
    win_rates = [fair_stats['win_rate'] * 100, tweaked_stats['win_rate'] * 100]
    theoretical_rates = [fair_game.p_win * 100, tweaked_game.p_win * 100]
    x = np.arange(len(names))
    width = 0.35
    ax3.bar(x - width/2, win_rates, width, label='Empirical', color='steelblue')
    ax3.bar(x + width/2, theoretical_rates, width, label='Theoretical', color='lightcoral')
    ax3.set_ylabel('Player Win Rate (%)')
    ax3.set_title('Player Win Rate: Empirical vs Theoretical')
    ax3.set_xticks(x)
    ax3.set_xticklabels(names)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Loss rate (House wins) comparison
    ax4 = axes[1, 1]
    loss_rates = [(1 - fair_stats['win_rate']) * 100, (1 - tweaked_stats['win_rate']) * 100]
    theoretical_loss = [(1 - fair_game.p_win) * 100, (1 - tweaked_game.p_win) * 100]
    ax4.bar(x - width/2, loss_rates, width, label='Empirical', color='darkgreen')
    ax4.bar(x + width/2, theoretical_loss, width, label='Theoretical', color='lightgreen')
    ax4.set_ylabel('House Win Rate (%)')
    ax4.set_title('House (Computer) Win Rate')
    ax4.set_xticks(x)
    ax4.set_xticklabels(names)
    ax4.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='50% Fair Line')
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
    
    return plots


# ============================================================================
# GAME STATE CLASS FOR INTERACTIVE GAME
# ============================================================================

class GameState:
    """
    Game state for interactive 2-player game.
    
    For tweaked mode (Mode 3/4):
    - even_prob: probability that bullet is in an EVEN chamber (Computer's turn)
    - odd_prob is automatically (1 - even_prob) since they MUST sum to 100%
    
    Player goes on ODD rounds (1,3,5), Computer goes on EVEN rounds (2,4,6).
    - If bullet is in odd chamber → Player gets shot → Player loses
    - If bullet is in even chamber → Computer gets shot → Player wins
    
    For house advantage: set even_prob < 0.5 (bullet more likely in player's chambers)
    """
    
    def __init__(self, player1_name, player2_name, chambers=6, bullets=1, 
                 even_prob=None, is_casino_mode=False, starting_balance=1000, initial_bet=100):
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.players = [player1_name, player2_name]
        self.current_player_index = 0  # Boy (Player 1) always goes first
        self.chambers = chambers
        self.bullets = bullets
        
        # Even/Odd probability system (must sum to 100%)
        self.even_prob = even_prob  # None = fair game, or 0.0-1.0 for tweaked
        self.odd_prob = (1.0 - even_prob) if even_prob is not None else None
        
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
        """
        Load the gun - determines bullet placement.
        
        For fair game (even_prob is None): bullet randomly placed in any chamber.
        For tweaked game: We determine WHO will get shot based on probability,
        then place the bullet in a chamber that ensures that person gets shot.
        
        Player (Boy) goes first, then Computer (Girl), alternating.
        Round 1,3,5... = Player's turn
        Round 2,4,6... = Computer's turn
        """
        self.current_chamber = 0  # Always start at chamber 0 for predictability
        self.chambers_checked = 0
        
        if self.even_prob is None:
            # Fair game: bullet equally likely in any chamber
            self.bullet_positions = random.sample(range(self.chambers), self.bullets)
        else:
            # Tweaked game: determine WHO gets shot based on probability
            # even_prob = probability Computer gets shot (player wins)
            # odd_prob = 1 - even_prob = probability Player gets shot (player loses)
            
            self.bullet_positions = []
            for _ in range(self.bullets):
                if random.random() < self.even_prob:
                    # Computer gets shot - put bullet in even chamber (0-indexed: 1,3,5)
                    # These are chambers 2,4,6 which are Computer's turns
                    even_chambers = [i for i in range(self.chambers) if i % 2 == 1]
                    available = [c for c in even_chambers if c not in self.bullet_positions]
                    if available:
                        self.bullet_positions.append(random.choice(available))
                    else:
                        # Fallback: use any available chamber
                        available = [c for c in range(self.chambers) if c not in self.bullet_positions]
                        if available:
                            self.bullet_positions.append(random.choice(available))
                else:
                    # Player gets shot - put bullet in odd chamber (0-indexed: 0,2,4)
                    # These are chambers 1,3,5 which are Player's turns
                    odd_chambers = [i for i in range(self.chambers) if i % 2 == 0]
                    available = [c for c in odd_chambers if c not in self.bullet_positions]
                    if available:
                        self.bullet_positions.append(random.choice(available))
                    else:
                        # Fallback: use any available chamber
                        available = [c for c in range(self.chambers) if c not in self.bullet_positions]
                        if available:
                            self.bullet_positions.append(random.choice(available))
    
    def pull_trigger(self):
        """
        Pull trigger and return result.
        
        Checks if current chamber has a bullet. The bullet placement was
        determined in load_gun() to favor either player or computer based
        on the probability settings.
        """
        if self.chambers_checked >= self.chambers:
            self.load_gun()
            return None  # Reload signal
        
        # Check if bullet is in current chamber
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
            'even_prob': self.even_prob,
            'odd_prob': self.odd_prob,
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
            data.get('even_prob'),  # Use even_prob, odd_prob auto-calculated
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
    is_casino_mode = game_state.is_casino_mode
    
    if game_state.round_number == 1 and game_state.image_phase == 'start':
        if is_casino_mode:
            return "computer vs player/first_image_gun_in_table.png"
        return "first_image_gun_in_table.png"
    
    if game_state.image_phase == 'getting_gun':
        if is_casino_mode:
            if current_player_is_boy:
                return "computer vs player/2nd_player_getting_gun.png"
            else:
                return "computer vs player/2nd_computer_getting_gun.png"
        else:
            if current_player_is_boy:
                return "2nd_boy_getting_gun.png"
            else:
                return "2nd_girl_getting_gun.png"
    elif game_state.image_phase == 'pointing':
        if is_casino_mode:
            if current_player_is_boy:
                return "computer vs player/3rd_gun_pointed_to_player.png"
            else:
                return "computer vs player/3rd_gun_pointed_to_computer.png"
        else:
            if current_player_is_boy:
                return "3rd_gun_pointed_to_boy.png"
            else:
                return "3rd_gun_pointed_to_girl.png"
    
    if is_casino_mode:
        return "computer vs player/first_image_gun_in_table.png"
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
        
        even_prob = None  # None = fair game
        is_casino_mode = False
        starting_balance = 1000
        initial_bet = 100
        
        if mode == '3' or mode == '4':
            # Get even_prob from form (0-100%), odd_prob is automatically 100% - even_prob
            even_prob_input = data.get('even_prob')
            if even_prob_input is not None:
                even_prob = float(even_prob_input) / 100
                # Validate that it's a valid probability (0-1)
                if even_prob < 0 or even_prob > 1:
                    return jsonify({'status': 'error', 'message': 'Even probability must be between 0 and 100%'})
        
        if mode == '4':
            is_casino_mode = True
            starting_balance = int(data.get('starting_balance', 1000))
            initial_bet = int(data.get('initial_bet', 100))
            player2_name = 'Computer 🤖'
        
        game = GameState(
            player1_name, player2_name, chambers, bullets,
            even_prob, is_casino_mode, starting_balance, initial_bet
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
    if game_state.even_prob is not None:
        # Display the even/odd probability info
        prob_info = f"Bullet Distribution: {game_state.even_prob * 100:.0f}% Even / {game_state.odd_prob * 100:.0f}% Odd"
        if game_state.even_prob < 0.5:
            prob_info += " (House Advantage)"
        elif game_state.even_prob > 0.5:
            prob_info += " (Player Advantage)"
    
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
        # Probability info for display
        'even_prob': game_state.even_prob,
        'odd_prob': game_state.odd_prob,
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
        
        # Use different images for casino mode (player vs computer)
        if game_state.is_casino_mode:
            image = f"computer vs player/4th_gun_shot_{'player' if current_player_is_boy else 'computer'}.png"
        else:
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
        # Safe outcome - use different images for casino mode
        if game_state.is_casino_mode:
            if current_player_is_boy:
                image = "computer vs player/4th_gun_shot_safe_bring_back_gun_to_table_player.png"
            else:
                image = "computer vs player/4th_gun_shot_safe_bring_back_gun_to_table_computer.png"
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
    even_prob_pct = float(data.get('even_prob', 35))  # Get as percentage (0-100)
    
    # Convert to decimal (0-1) and validate
    even_prob = even_prob_pct / 100.0
    even_prob = max(0.0, min(1.0, even_prob))  # Clamp to 0-1
    
    # Run simulation with even_prob
    simulation_data = run_monte_carlo_simulation(num_plays, bet_amount, even_prob)
    
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
            'empirical_win_rate': round(fair_stats['empirical_win_rate'] * 100, 2),
            'theoretical_win_rate': round(fair_stats['theoretical_win_rate'] * 100, 2),
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
            'empirical_win_rate': round(tweaked_stats['empirical_win_rate'] * 100, 2),
            'theoretical_win_rate': round(tweaked_stats['theoretical_win_rate'] * 100, 2),
            'p_win': round(tweaked_game.p_win * 100, 2),
            'p_lose': round(tweaked_game.p_lose * 100, 2),
            'even_prob': round(tweaked_game.even_prob * 100, 2),
            'odd_prob': round(tweaked_game.odd_prob * 100, 2),
        }
    })


if __name__ == '__main__':
    app.run(debug=False)
