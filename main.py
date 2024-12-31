import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import random

class NBAPlayoffPredictor:
    def __init__(self):
        self.model = None
        self.games = None
        self.rankings = None
        self.team_records = {}
        self.team_point_diff = {}

    def prepare_model(self, games_file, ranking_file):
        print("Loading data and preparing model...")
        self.games = pd.read_csv(games_file)
        self.games = self.games[
            (self.games['GAME_DATE_EST'] >= '2018-10-01') & 
            (self.games['GAME_DATE_EST'] <= '2019-04-30')
        ]
        
        X, y = [], []
        self.team_records = {}
        self.team_point_diff = {}

        for i, game in self.games.iterrows():
            home_id, away_id = game['HOME_TEAM_ID'], game['VISITOR_TEAM_ID']
            self.team_records.setdefault(home_id, {'home_wins': 0, 'home_games': 0, 'away_wins': 0, 'away_games': 0})
            self.team_records.setdefault(away_id, {'home_wins': 0, 'home_games': 0, 'away_wins': 0, 'away_games': 0})
            self.team_point_diff.setdefault(home_id, []).append(game['PTS_home'] - game['PTS_away'])
            self.team_point_diff.setdefault(away_id, []).append(game['PTS_away'] - game['PTS_home'])
            self.team_records[home_id]['home_games'] += 1
            self.team_records[away_id]['away_games'] += 1
            if game['HOME_TEAM_WINS'] == 1:
                self.team_records[home_id]['home_wins'] += 1
            else:
                self.team_records[away_id]['away_wins'] += 1

        for i, game in self.games.iterrows():
            home_id, away_id = game['HOME_TEAM_ID'], game['VISITOR_TEAM_ID']
            if home_id in self.team_records and away_id in self.team_records and home_id in self.team_point_diff and away_id in self.team_point_diff:
                home_stats = [
                    game['PTS_home'], 
                    game['FG_PCT_home'], 
                    game['FT_PCT_home'], 
                    game['FG3_PCT_home'], 
                    game['AST_home'], 
                    game['REB_home'],
                    self.team_records[home_id]['home_wins'] / max(1, self.team_records[home_id]['home_games']),
                    sum(self.team_point_diff[home_id]) / max(1, len(self.team_point_diff[home_id]))
                ]
                away_stats = [
                    game['PTS_away'], 
                    game['FG_PCT_away'], 
                    game['FT_PCT_away'], 
                    game['FG3_PCT_away'], 
                    game['AST_away'], 
                    game['REB_away'],
                    self.team_records[away_id]['away_wins'] / max(1, self.team_records[away_id]['away_games']),
                    sum(self.team_point_diff[away_id]) / max(1, len(self.team_point_diff[away_id]))
                ]
                X.append(home_stats + away_stats)
                y.append(game['HOME_TEAM_WINS'])

        X = np.array(X)
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X_train, y_train)
        print("Model training complete.")

        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        print(f"Model Accuracy: {accuracy:.2f}")
        print(f"Model ROC-AUC: {roc_auc:.2f}")

    def simulate_game(self, home_stats, away_stats):
        features = home_stats + away_stats
        prob = self.model.predict_proba([features])[0][1]
        return 1 if random.random() < prob else 0

    def run_tournament_simulation(self, playoff_teams, simulations=1000):
        print(f"\nRunning {simulations} tournament simulations...")
        championship_counts = {team: 0 for team, _, _ in playoff_teams}

        for sim in range(simulations):
            teams = playoff_teams.copy()
            random.shuffle(teams)

            while len(teams) > 1:
                next_round = []
                for i in range(0, len(teams), 2):
                    if i+1 >= len(teams):
                        next_round.append(teams[i])
                        continue
                    team1, team2 = teams[i], teams[i+1]
                    team1_name, team1_id, team1_wins = team1
                    team2_name, team2_id, team2_wins = team2

                    team1_games = self.games[
                        (self.games['HOME_TEAM_ID'] == team1_id) | 
                        (self.games['VISITOR_TEAM_ID'] == team1_id)
                    ]
                    team2_games = self.games[
                        (self.games['HOME_TEAM_ID'] == team2_id) | 
                        (self.games['VISITOR_TEAM_ID'] == team2_id)
                    ]

                    if team1_games.empty or team2_games.empty:
                        winner = team1 if random.random() < 0.5 else team2
                        next_round.append(winner)
                        continue

                    team1_home = team1_games[team1_games['HOME_TEAM_ID'] == team1_id]
                    team1_away = team1_games[team1_games['VISITOR_TEAM_ID'] == team1_id]
                    team2_home = team2_games[team2_games['HOME_TEAM_ID'] == team2_id]
                    team2_away = team2_games[team2_games['VISITOR_TEAM_ID'] == team2_id]

                    team1_avg_stats = [
                        team1_home['PTS_home'].mean(), 
                        team1_home['FG_PCT_home'].mean(), 
                        team1_home['FT_PCT_home'].mean(), 
                        team1_home['FG3_PCT_home'].mean(), 
                        team1_home['AST_home'].mean(), 
                        team1_home['REB_home'].mean(),
                        self.team_records[team1_id]['home_wins'] / max(1, self.team_records[team1_id]['home_games']),
                        sum(self.team_point_diff[team1_id]) / max(1, len(self.team_point_diff[team1_id]))
                    ]

                    team2_avg_stats = [
                        team2_home['PTS_home'].mean(), 
                        team2_home['FG_PCT_home'].mean(), 
                        team2_home['FT_PCT_home'].mean(), 
                        team2_home['FG3_PCT_home'].mean(), 
                        team2_home['AST_home'].mean(), 
                        team2_home['REB_home'].mean(),
                        self.team_records[team2_id]['home_wins'] / max(1, self.team_records[team2_id]['home_games']),
                        sum(self.team_point_diff[team2_id]) / max(1, len(self.team_point_diff[team2_id]))
                    ]

                    game_result = self.simulate_game(team1_avg_stats, team2_avg_stats)
                    winner = team1 if game_result == 1 else team2
                    next_round.append(winner)
                
                teams = next_round

            champion = teams[0][0]
            championship_counts[champion] += 1

        championship_probabilities = {team: (count / simulations) * 100 for team, count in championship_counts.items()}

        print("\nChampionship probabilities:")
        sorted_teams = sorted(championship_probabilities.items(), key=lambda item: item[1], reverse=True)
        for team, prob in sorted_teams:
            print(f"{team}: {prob:.1f}%")

        self.visualize_probabilities(championship_probabilities)

    def visualize_probabilities(self, team_probabilities):
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.set_facecolor('#2F2F2F')
        fig.patch.set_facecolor('#2F2F2F')
        
        team_probs_list = list(team_probabilities.items())
        team_probs_list.sort(reverse=True, key=lambda item: item[1])
        sorted_teams = team_probs_list
        teams = [team for team, _ in sorted_teams]
        probs = [prob for _, prob in sorted_teams]
        colors = ['#FF0000', '#4CAF50', '#2196F3', '#9C27B0', '#FF9800'] * 3
        bars = ax.bar(range(len(teams)), probs, color=colors[:len(teams)])
        
        for i in range(len(bars)):
            bar = bars[i]
            x_position = bar.get_x() + bar.get_width() / 2
            y_position = bar.get_height()
            label = f'{probs[i]:.1f}%'
            ax.text(x_position, y_position, label, ha='center', va='bottom', color='white', fontweight='bold')
        
        ax.set_title('NBA Championship Probability by Team', pad=20, color='white')
        ax.set_ylabel('Probability (%)', color='white')
        ax.set_xticks(range(len(teams)))
        ax.set_xticklabels(teams, rotation=45, ha='right', color='white')
        ax.grid(True, alpha=0.1)
        plt.subplots_adjust(right=0.85)
        plt.savefig('probabilities.png', bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == "__main__":
    predictor = NBAPlayoffPredictor()
    predictor.prepare_model('data/games.csv', 'data/ranking.csv')
    predictor.run_tournament_simulation([
        ('Milwaukee Bucks', 1610612749, 60),      
        ('Toronto Raptors', 1610612761, 58),      
        ('Golden State Warriors', 1610612744, 57), 
        ('Denver Nuggets', 1610612743, 54),       
        ('Portland Trail Blazers', 1610612757, 53),
        ('Houston Rockets', 1610612745, 53),
        ('Philadelphia 76ers', 1610612755, 51),
        ('Utah Jazz', 1610612762, 50),
        ('Boston Celtics', 1610612738, 49),
        ('Oklahoma City Thunder', 1610612760, 49),
        ('Indiana Pacers', 1610612754, 48),
        ('San Antonio Spurs', 1610612759, 48),
        ('LA Clippers', 1610612746, 48),
        ('Brooklyn Nets', 1610612751, 42),
        ('Orlando Magic', 1610612753, 42),
        ('Detroit Pistons', 1610612765, 41)
    ], simulations=1000)
