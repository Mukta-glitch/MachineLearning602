show databases;
use baseball;


drop 
  table IF EXISTS home_team_wins;

CREATE table home_team_wins AS 
SELECT 
  pc.game_id, 
  pc.team_id,
  CASE WHEN b.away_runs < b.home_runs THEN 1 WHEN b.away_runs > b.home_runs THEN 0 ELSE 0 END AS home_team_wins 
FROM 
  boxscore b 
JOIN 
  pitcher_counts pc 
ON 
  pc.game_id = b.game_id 
GROUP BY 
  pc.team_id;

CREATE INDEX index_game_id ON home_team_wins(game_id);

describe home_team_wins;

drop 
  table IF EXISTS joined;

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

#drop table if exists rolling;

CREATE TEMPORARY TABLE team_avg AS
SELECT 
  team_id, 
  local_date,
  AVG(Slugging_Percentage) AS avg_Slugging_Percentage,
  AVG(away_wp) AS Avg_away_wp,
  AVG(home_wp) AS Avg_home_wp,
  AVG(HomeRuns_per_9_Innings) As Avg_HomeRuns_per_9_innings,
  AVG(Batting_Average) AS Avg_batting_average,
  AVG(On_Base_Percentage) As Avg_On_Base_Percentage,
  AVG(Run_Differential) As Avg_Run_Differential,
  AVG(Times_on_Base) As Avg_Times_on_Base,
  AVG(Earned_Run_Average) As Avg_Earned_Run_Average,
  AVG(innings_pitched) As Avg_innings_pitched,
  Avg(Walk_plus_Hits_per_Inning_Pitched) As Avg_Walk_plus_Hits_per_Inning_Pitched,
  Avg(Strike_To_Walk_Ratio) As Avg_Strike_To_Walk_Ratio
FROM 
  joined
GROUP BY 
  team_id, 
  local_date;

drop 
  table IF EXISTS rolling;

CREATE TABLE rolling AS 
SELECT 
  t1.team_id,
  t1.local_date,
  AVG(t2.avg_Slugging_Percentage) AS rolling_Slugging_Percentage,
  AVG(t2.Avg_innings_pitched) AS rolling_innings_pitched,
  AVG(t2.Avg_away_wp) As rolling_away_wp,
  AVG(t2.Avg_home_wp) As rolling_home_wp,
  AVG(t2.Avg_Strike_To_Walk_Ratio) As rolling_Strike_To_Walk_Ratio,
  AVG(t2.Avg_Walk_plus_Hits_per_Inning_Pitched) As rolling_Walk_plus_Hits_per_Inning_Pitched,
  AVG(t2.Avg_HomeRuns_per_9_Innings) As rolling_HomeRuns_per_9_Innings,
  AVG(t2.Avg_Batting_Average) As rolling_Batting_Average,
  AVG(t2.Avg_Times_on_Base) As rolling_Times_On_Base,
  AVG(t2.Avg_Earned_Run_Average) As rolling_Earned_Run_Average,
  AVG(t2.Avg_On_Base_Percentage) As rolling_On_Base_Percentage,
  AVG(t2.Avg_Run_Differential) As rolling_run_differential
FROM 
  team_avg AS t1 
  JOIN team_avg AS t2 ON (
    t1.team_id = t2.team_id AND 
    t2.local_date BETWEEN DATE_SUB(t1.local_date, INTERVAL 100 DAY) AND t1.local_date
  )
GROUP BY 
  t1.team_id;

drop table if exists joined1;
create table joined1 as select 
  r.team_id,r.rolling_away_wp,r.rolling_run_differential,r.rolling_home_wp,r.rolling_Batting_Average,r.rolling_innings_pitched,
  r.rolling_Slugging_Percentage,r.rolling_Times_On_Base,r.rolling_Earned_Run_Average,r.rolling_On_Base_Percentage,r.rolling_HomeRuns_per_9_Innings,
  r.rolling_Walk_plus_Hits_per_Inning_Pitched,r.rolling_Strike_To_Walk_Ratio,h.home_team_wins
 from rolling r join home_team_wins h on r.team_id=h.team_id;
select * from joined1;
#select * from rolling;

