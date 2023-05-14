#show databases;
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
  b.away_runs, 
  pc.Sac_Fly, 
  pc.Fan_Interference,
  b.home_runs, 
  pc.stolenBase2B,
  pc.stolenBase3B,
  pc.stolenBaseHome,
  pc.caughtStealing2B,
  pc.caughtStealing3B,
  pc.caughtStealingHome,
  pc.Sacrifice_Bunt_DP,
  pc.Field_Error,
  pc.Batter_Interference,
  pc.Catcher_Interference,
  pc.Grounded_Into_DP,
  COALESCE((pc.Strikeout + pc.Walk)/NULLIF((pc.endingInning - pc.startingInning), 0)) AS Power_Finesse_Ratio,
  COALESCE(pc.Strikeout / NULLIF(pc.Walk, 0),  0) AS Strike_To_Walk_Ratio, 
  COALESCE(pc.Walk + (bc.Hit / NULLIF( ( pc.endingInning - pc.startingInning ),  0 )),  0) AS Walk_plus_Hits_per_Inning_Pitched, 
  COALESCE(9 * (b.away_runs / NULLIF((pc.endingInning - pc.startingInning ), 0) ), 0) AS Earned_Run_Average, 
  (bc.Hit + pc.Walk + bc.Hit_By_Pitch) AS Times_on_Base, 
(pc.endingInning - pc.startingInning) As innings_pitched, 
  (b.home_hits - b.away_hits) As Run_Differential, 
  (bc.Hit + pc.Walk + bc.Hit_By_Pitch) / NULLIF((bc.atBat + pc.Walk + bc.Hit_By_Pitch + pc.Sac_Fly), 0) AS On_Base_Percentage, 
  bc.Hit / NULLIF(bc.atBat, 0) AS Batting_Average, 
 COALESCE(9 * (b.home_runs / NULLIF((pc.endingInning - pc.startingInning), 0)), 0) AS HomeRuns_per_9_Innings, 
  (g.home_w) / NULLIF( ((g.home_w) + (g.home_l)),  0) AS home_wp, 
  (g.away_w) / NULLIF(((g.away_w) + (g.away_l)), 0) AS away_wp, 
    (pc.stolenBase2B + pc.stolenBase3B + pc.stolenBaseHome) AS totalStolenBases,
    (pc.caughtStealing2B + pc.caughtStealing3B + pc.caughtStealingHome) AS totalCaughtStealing,
  CASE WHEN bc.atBat = 0 THEN NULL ELSE ((bc.Single) + 2 *(bc.Double) + 3 *(bc.Triple) + 4 *(bc.Home_Run)) / (bc.atBat) END AS Slugging_Percentage,
CASE 
  WHEN CAST(REPLACE(b.temp, ' degrees', '') AS UNSIGNED) >= 60 AND CAST(REPLACE(b.temp, ' degrees', '') AS UNSIGNED) <= 80 THEN 0
  WHEN CAST(REPLACE(b.temp, ' degrees', '') AS UNSIGNED) < 60 THEN -1
  WHEN CAST(REPLACE(b.temp, ' degrees', '') AS UNSIGNED) > 80 THEN 1
  ELSE -2
END AS temp_condition,

CASE 
    WHEN wind LIKE '%mph' THEN 
      CASE 
        WHEN CAST(SUBSTRING_INDEX(b.wind, ' ', 1) AS UNSIGNED) <= 10 THEN -1
        WHEN CAST(SUBSTRING_INDEX(b.wind, ' ', 1) AS UNSIGNED) > 10 AND CAST(SUBSTRING_INDEX(b.wind, ' ', 1) AS UNSIGNED) <= 15 THEN 0
        WHEN CAST(SUBSTRING_INDEX(b.wind, ' ', 1) AS UNSIGNED) > 15 THEN 1
        ELSE -2
      END
    ELSE -2
  END AS wind_condition,
ROUND(((nullif(bc.Home_Run,0) / bc.Hit) / nullif((nullif(bc.Home_Run,0) / bc.Hit),0)),5) AS Home_Run_to_Hit_Ratio,
b.home_errors-b.away_errors AS Error_differential_ratio  
FROM 
  batter_counts bc 
  left JOIN pitcher_counts pc ON bc.game_id = pc.game_id 
  AND bc.team_id = pc.team_id 
  left JOIN game g ON g.game_id = pc.game_id 
  AND g.game_id = bc.game_id 
  left JOIN boxscore b On b.game_id = pc.game_id 
  AND b.game_id = g.game_id;
describe joined;
CREATE INDEX index_team_id on joined(team_id);
drop table if exists rolling;
drop 
  table IF EXISTS team_avg;
CREATE TEMPORARY TABLE team_avg AS
SELECT 
  team_id, 
  local_date,
  AVG(Power_Finesse_Ratio) AS avg_Power_Finesse_Ratio,
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
  Avg(Strike_To_Walk_Ratio) As Avg_Strike_To_Walk_Ratio,
  AVG(temp_condition) as Avg_Tc,
  AVG(wind_condition) as Avg_WC,
  AVG(totalStolenBases) As Avg_Total_Stolen_Bases,
  AVG(totalCaughtStealing) as Avg_Total_Caught_Stealing,
  AVG(Fan_Interference) as Avg_Fan_Interference,
  AVG(Field_Error) AS Avg_Field_Error,
  AVG(Batter_Interference) As Avg_Batter_Interference,
  AVG(Catcher_Interference) AS Avg_Catcher_Interference,
  AVG(Grounded_Into_DP) As Avg_Grounded_Into_DP,
  AVG(Sac_Fly) As Avg_Sac_Fly,
  AVG(Error_differential_ratio) As Avg_Error_differential_ratio,
  AVG(Sacrifice_Bunt_DP) As Avg_Sacrifice_Bunt_DP
FROM 
  joined
GROUP BY 
  team_id, 
  local_date;
select * from team_avg;
drop 
  table IF EXISTS rolling;
CREATE TABLE rolling AS 
SELECT 
  t1.team_id,
  t1.local_date,
  AVG(t2.avg_Power_Finesse_Ratio) as rolling_Power_Finesse_Ratio,
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
  AVG(t2.Avg_Run_Differential) As rolling_run_differential,
  AVG(t2.Avg_Tc) As temperature,
  AVG(t2.Avg_WC) As wind,
  AVG(t2.Avg_Grounded_Into_DP) as rolling_Grounded_Into_DP,
  AVG(t2.Avg_Fan_Interference) as rolling_Fan_Interference,
  (t2.Avg_Field_Error) as rolling_Field_Error,
  AVG(t2.Avg_Batter_Interference) as rollingBI,
 AVG(t2.Avg_Catcher_Interference) as rollingCI,
 AVG(t2.Avg_Sac_Fly) as rollingSF,
 AVG(t2.Avg_Error_differential_ratio) as rolling_Error_differential_ratio,
 AVG(t2.Avg_Sacrifice_Bunt_DP) as rolling_Sacrifice_Bunt_DP  
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
r.team_id,r.rolling_away_wp,r.rolling_run_differential,r.rolling_home_wp,r.rolling_Batting_Average,r.rolling_innings_pitched,    		r.rolling_Slugging_Percentage,r.rolling_Times_On_Base,r.rolling_Error_differential_ratio,
r.rolling_Earned_Run_Average,r.rolling_On_Base_Percentage,r.rolling_HomeRuns_per_9_Innings,
r.rolling_Sacrifice_Bunt_DP,
r.rolling_Walk_plus_Hits_per_Inning_Pitched,r.rolling_Strike_To_Walk_Ratio,h.home_team_wins,r.temperature,r.wind,
r.rolling_Fan_Interference,r.rolling_Field_Error,r.rollingBI,r.rollingCI,r.rolling_Grounded_Into_DP,r.rollingSF,
r.rolling_Power_Finesse_Ratio
 from rolling r join home_team_wins h on r.team_id=h.team_id;
select * from joined1 limit 10;
#select * from rolling limit 10;
