show databases;
use baseball;
create table battingaverage as 
select 
  B.game_id, 
  B.batter, 
  DATE(B.updatedDate) as ud, 
  Case when SUM(B.atBat)!= 0 then SUM(B.hit) / SUM(B.atBat) else 0 end as Batting_average 
from 
  batter_counts B 
  join batter_counts G on B.game_id = G.game_id 
group by 
  DATE(B.updatedDate), 
  B.batter 
order by 
  B.batter, 
  DATE(B.updatedDate) desc;
  
SELECT batter,avg(Batting_average),YEAR(ud) AS Historic_batting_avg
FROM battingaverage
GROUP BY batter order by YEAR(ud);

CREATE INDEX batterindex ON battingaverage(batter);

select 
  * 
from 
  battingaverage limit 10;
  Create table rolling as 
SELECT batter, ud, 
       AVG(Batting_average) OVER (PARTITION BY batter ORDER BY ud) AS rolling_average
FROM battingaverage
WHERE batter = batter AND ud BETWEEN DATE_SUB(ud, INTERVAL 100 DAY) AND ud
ORDER BY ud DESC;

select * from rolling;
drop 
  table battingaverage;
drop 
  table rolling;


