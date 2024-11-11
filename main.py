import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

class NBAPlayoffPredictor:
    def __init__(self):
        self.model = None
        self.games = None
        self.rankings = None

    def prepare_model(self, games_file, ranking_file):
        print("Loading data and preparing model...")
        self.games = pd.read_csv(games_file)
        self.games = self.games[(self.games['GAME_DATE_EST'] >= '2018-10-01') & (self.games['GAME_DATE_EST'] <= '2019-04-30')]
        
        X, y = [], []
        team_records, team_point_diff = {}, {}

        for _, game in self.games.iterrows():
            home_id, away_id = game['HOME_TEAM_ID'], game['VISITOR_TEAM_ID']
            team_records.setdefault(home_id, {'home_wins': 0, 'home_games': 0, 'away_wins': 0, 'away_games': 0})
            team_records.setdefault(away_id, {'home_wins': 0, 'home_games': 0, 'away_wins': 0, 'away_games': 0})
            team_point_diff.setdefault(home_id, []).append(game['PTS_home'] - game['PTS_away'])
            team_point_diff.setdefault(away_id, []).append(game['PTS_away'] - game['PTS_home'])
            team_records[home_id]['home_games'] += 1
            team_records[away_id]['away_games'] += 1
            if game['HOME_TEAM_WINS'] == 1:
                team_records[home_id]['home_wins'] += 1
            else:
                team_records[away_id]['away_wins'] += 1

        for _, game in self.games.iterrows():
            home_id, away_id = game['HOME_TEAM_ID'], game['VISITOR_TEAM_ID']
            if home_id in team_records and away_id in team_records and home_id in team_point_diff and away_id in team_point_diff:
                home_stats = [
                    game['PTS_home'], game['FG_PCT_home'], game['FT_PCT_home'], game['FG3_PCT_home'], 
                    game['AST_home'], game['REB_home'],
                    team_records[home_id]['home_wins'] / max(1, team_records[home_id]['home_games']),
                    sum(team_point_diff[home_id]) / max(1, len(team_point_diff[home_id]))
                ]
                away_stats = [
                    game['PTS_away'], game['FG_PCT_away'], game['FT_PCT_away'], game['FG3_PCT_away'], 
                    game['AST_away'], game['REB_away'],
                    team_records[away_id]['away_wins'] / max(1, team_records[away_id]['away_games']),
                    sum(team_point_diff[away_id]) / max(1, len(team_point_diff[away_id]))
                ]
                X.append(home_stats + away_stats)
                y.append(game['HOME_TEAM_WINS'])

        self.model = LogisticRegression()
        self.model.fit(np.array(X), np.array(y))
        print("Model training complete.")

    def predict_championship_prob(self, team_id, wins):
        team_games = self.games[
            (self.games['HOME_TEAM_ID'] == team_id) | 
            (self.games['VISITOR_TEAM_ID'] == team_id)
        ]
        
        if not team_games.empty:
            home_games = team_games[team_games['HOME_TEAM_ID'] == team_id]
            away_games = team_games[team_games['VISITOR_TEAM_ID'] == team_id]
            if not home_games.empty and not away_games.empty:
                home_diffs = home_games['PTS_home'] - home_games['PTS_away']
                away_diffs = away_games['PTS_away'] - away_games['PTS_home']
                avg_point_diff = (home_diffs.mean() + away_diffs.mean()) / 2
                
                home_win_pct = home_games['HOME_TEAM_WINS'].sum() / max(1, len(home_games))
                away_win_pct = (len(away_games) - away_games['HOME_TEAM_WINS'].sum()) / max(1, len(away_games))
                
                avg_stats = [
                    (home_games['PTS_home'].mean() + away_games['PTS_away'].mean()) / 2,
                    (home_games['FG_PCT_home'].mean() + away_games['FG_PCT_away'].mean()) / 2,
                    (home_games['FT_PCT_home'].mean() + away_games['FT_PCT_away'].mean()) / 2,
                    (home_games['FG3_PCT_home'].mean() + away_games['FG3_PCT_away'].mean()) / 2,
                    (home_games['AST_home'].mean() + away_games['AST_away'].mean()) / 2,
                    (home_games['REB_home'].mean() + away_games['REB_away'].mean()) / 2,
                    (home_win_pct + away_win_pct) / 2,
                    avg_point_diff
                ]
                
                features = avg_stats + avg_stats
                base_prob = self.model.predict_proba([features])[0][1]
                win_scale = wins / 82
                diff_scale = (avg_point_diff + 10) / 20

                if wins >= 57:
                    return base_prob * 100 * win_scale * diff_scale
                elif wins >= 53:
                    return base_prob * 70 * win_scale * diff_scale
                elif wins >= 49:
                    return base_prob * 50 * win_scale * diff_scale
                else:
                    return base_prob * 30 * win_scale * diff_scale
        return 5

    def run_tournament(self, teams_file):
        print("\nCalculating championship probabilities...")
        playoff_teams = [
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
        ]

        team_probabilities = {}
        for team, team_id, wins in playoff_teams:
            team_probabilities[team] = self.predict_championship_prob(team_id, wins)

        total = sum(team_probabilities.values())
        for team in team_probabilities:
            team_probabilities[team] = (team_probabilities[team] / total) * 100

        print("\nChampionship probabilities:")
        team_probs_list = list(team_probabilities.items())
        team_probs_list.sort(reverse=True, key=lambda item: item[1])
        sorted_teams = team_probs_list
        for team, prob in sorted_teams:
            print(f"{team}: {prob:.1f}%")

        self.visualize_probabilities(team_probabilities)


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
    predictor.run_tournament('data/teams.csv')
