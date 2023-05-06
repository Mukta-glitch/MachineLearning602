CREATE OR REPLACE TEMPORARY TABLE sum_batter_counts AS
SELECT
  game_id,
  batter,
  SUM(hit) AS hit,
  SUM(atbat) AS at_bat
FROM batter_counts
GROUP BY game_id, batter;

CREATE INDEX batter_counts_temp_game_id_batter_idx ON sum_batter_counts (game_id, batter);

select * from sum_batter_counts;

CREATE OR REPLACE TEMPORARY TABLE rolling1 AS
SELECT
   g.game_id,
   bc.batter AS batter,
   bc.hit,
   bc.at_bat,
   g.local_date
FROM batter_counts_temp AS bc
INNER JOIN game g ON bc.game_id = game.game_id;

CREATE INDEX batter_avg_rolling_temp_batter_idx ON rolling1 (batter);

select * from rolling1;

CREATE OR REPLACE TABLE rolling2 AS
SELECT
  b1.batter,
  (
    CASE
      WHEN SUM(b2.at_bat) > 0
      THEN SUM(b2.hit) / SUM(b2.at_bat)
      ELSE 0
    END
  ) AS batting_avg,
  b1.game_id,
FROM batter_avg_rolling_temp b1
INNER JOIN batter_avg_rolling_temp b2
ON b1.batter = b2.batter
AND b2.local_date < b1.local_date
AND b2.local_date > DATE_SUB(b1.local_date, INTERVAL 100 DAY)
WHERE bart1.game_id = 12560
GROUP BY b1.batter, b1.local_date
ORDER BY b1.batter;

CREATE INDEX batter_avg_rolling_temp_local_date_idx ON rolling2 (local_date);

select * from rolling2 ;
