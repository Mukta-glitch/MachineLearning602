show databases;
use baseball;


drop 
  table IF EXISTS home_team_wins;

-- The home_team_wins column will be 1 if the home team wins, 0 if the away team wins, and NULL if the game was a tie

CREATE table home_team_wins AS 
SELECT 
  game_id, 
  CASE WHEN away_runs < home_runs THEN 1 WHEN away_runs > home_runs THEN 0 ELSE 0 END AS home_team_wins 
FROM 
  boxscore;
-- Create an index on the game_id column of the home_team_wins table for faster querying
CREATE INDEX index_game_id ON home_team_wins(game_id);

describe home_team_wins;

drop 
  table IF EXISTS joined;
-- Drop the joined table if it exists, then create a new one with various columns from multiple tables
CREATE table joined AS 
SELECT 
  i.inning_id, 
  bc.team_id, 
  g.local_date, 
  bc.Hit, 
  bc.atBat, 
  bc.homeTeam, 
  bc.awayTeam, 
  bc.Home_Run, 
  bc.Hit_By_Pitch, 
  bc.Single, 
  bc.Double, 
  bc.Triple, 
  pc.startingPitcher, 
  pc.Walk, 
  pc.Strikeout, 
  pc.startingInning, 
  pc.endingInning, 
  pc.pitcher, 
  g.home_w, 
  g.home_l, 
  g.away_w, 
  g.away_l, 
  g.home_pitcher, 
  g.away_pitcher, 
  i.home_score, 
  i.game_id, 
  i.on_1b, 
  i.on_2b, 
  i.on_3b, 
  b.away_runs, 
  i.away_score, 
  pc.Sac_Fly, 
  b.home_runs, 
  COALESCE(
    pc.Strikeout / NULLIF(pc.Walk, 0), 
    0
  ) AS Strike_To_Walk_Ratio, 
  COALESCE(
    pc.Walk + (
      bc.Hit / NULLIF(
        (
          pc.endingInning - pc.startingInning
        ), 
        0
      )
    ), 
    0
  ) AS Walk_plus_Hits_per_Inning_Pitched, 
  -- Calculate various derived columns and give them aliases
  COALESCE(
    9 * (
      b.away_runs / NULLIF(
        (
          pc.endingInning - pc.startingInning
        ), 
        0
      )
    ), 
    0
  ) AS Earned_Run_Average, 
  (bc.Hit + pc.Walk + bc.Hit_By_Pitch) AS Times_on_Base, 
  (
    pc.endingInning - pc.startingInning
  ) As innings_pitched, 
  (i.home_score - i.away_score) As Run_Differential, 
  (bc.Hit + pc.Walk + bc.Hit_By_Pitch) / NULLIF(
    (
      bc.atBat + pc.Walk + bc.Hit_By_Pitch + pc.Sac_Fly
    ), 
    0
  ) AS On_Base_Percentage, 
  bc.Hit / NULLIF(bc.atBat, 0) AS Batting_Average, 
  COALESCE(
    9 * (
      b.home_runs / NULLIF(
        (
          pc.endingInning - pc.startingInning
        ), 
        0
      )
    ), 
    0
  ) AS HomeRuns_per_9_Innings, 
  (g.home_w) / NULLIF(
    (
      (g.home_w) + (g.home_l)
    ), 
    0
  ) AS home_wp, 
  (g.away_w) / NULLIF(
    (
      (g.away_w) + (g.away_l)
    ), 
    0
  ) AS away_wp, 
  CASE WHEN bc.atBat = 0 THEN NULL ELSE (
    (bc.Single) + 2 *(bc.Double) + 3 *(bc.Triple) + 4 *(bc.Home_Run)
  ) / (bc.atBat) END AS Slugging_Percentage 
FROM 
  batter_counts bc 
  left JOIN pitcher_counts pc ON bc.game_id = pc.game_id 
  AND bc.team_id = pc.team_id 
  left JOIN game g ON g.game_id = pc.game_id 
  AND g.game_id = bc.game_id 
  JOIN inning i On i.game_id = pc.game_id 
  AND i.game_id = g.game_id 
  left JOIN boxscore b On b.game_id = pc.game_id 
  AND b.game_id = g.game_id;

describe joined;

CREATE INDEX index_team_id on joined(team_id);

drop table if exists rolling;

CREATE TABLE rolling as 
SELECT 
  team_id, 
  AVG(Strike_To_Walk_Ratio) OVER (
    PARTITION BY team_id 
    ORDER BY 
      local_date ROWS BETWEEN 9 PRECEDING 
      AND CURRENT ROW
  ) AS rolling_Strike_To_Walk_Ratio, 
  AVG(
    Walk_plus_Hits_per_Inning_Pitched
  ) OVER (
    PARTITION BY team_id 
    ORDER BY 
      local_date ROWS BETWEEN 9 PRECEDING 
      AND CURRENT ROW
  ) AS rolling_Walk_plus_Hits_per_Inning_Pitched, 
  AVG(Earned_Run_Average) OVER (
    PARTITION BY team_id 
    ORDER BY 
      local_date ROWS BETWEEN 9 PRECEDING 
      AND CURRENT ROW
  ) AS rolling_Earned_Run_Average, 
  AVG(Slugging_Percentage) OVER (
    PARTITION BY team_id 
    ORDER BY 
      local_date ROWS BETWEEN 9 PRECEDING 
      AND CURRENT ROW
  ) AS rolling_Slugging_Percentage, 
  AVG(Times_on_Base) OVER (
    PARTITION BY team_id 
    ORDER BY 
      local_date ROWS BETWEEN 9 PRECEDING 
      AND CURRENT ROW
  ) AS rolling_Times_on_Base, 
  AVG(Run_Differential) OVER (
    PARTITION BY team_id 
    ORDER BY 
      local_date ROWS BETWEEN 9 PRECEDING 
      AND CURRENT ROW
  ) AS rolling_Run_Differential, 
  AVG(innings_pitched) OVER (
    PARTITION BY team_id 
    ORDER BY 
      local_date ROWS BETWEEN 9 PRECEDING 
      AND CURRENT ROW
  ) AS rolling_innings_pitched, 
  AVG(On_Base_Percentage) OVER (
    PARTITION BY team_id 
    ORDER BY 
      local_date ROWS BETWEEN 9 PRECEDING 
      AND CURRENT ROW
  ) AS rolling_On_Base_Percentage, 
  AVG(Batting_Average) OVER (
    PARTITION BY team_id 
    ORDER BY 
      local_date ROWS BETWEEN 9 PRECEDING 
      AND CURRENT ROW
  ) AS rolling_Batting_Average, 
  AVG(home_wp) OVER (
    PARTITION BY team_id 
    ORDER BY 
      local_date ROWS BETWEEN 9 PRECEDING 
      AND CURRENT ROW
  ) AS rolling_home_wp, 
  AVG(away_wp) OVER (
    PARTITION BY team_id 
    ORDER BY 
      local_date ROWS BETWEEN 9 PRECEDING 
      AND CURRENT ROW
  ) AS rolling_away_wp 
FROM 
  joined 
WHERE 
  local_date BETWEEN DATE_SUB(local_date, INTERVAL 100 DAY) 
  AND local_date 
group by 
  team_id;

  select * from rolling;
