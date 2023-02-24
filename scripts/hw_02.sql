show databases;
use baseball;
--calculate batting_average
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

--Index creation  
create INDEX batterindex on battingaverage(batter);

--Calculate Historic Batting Average
Create table historic as
select batter,avg(Batting_average),YEAR(ud) AS Historic_batting_avg
from battingaverage
group by batter order by YEAR(ud);

--to refer column names and values
select 
  * 
from 
  battingaverage limit 10;

--rolling average
Create table rolling as 
Select batter, ud, 
       AVG(Batting_average) OVER (PARTITION by batter order by ud) as rolling_average
From battingaverage
Where batter = batter and ud between DATE_SUB(ud, INTERVAL 100 DAY) and ud
Order by batter,ud;

select * from rolling;

--drop tables created
drop 
  table battingaverage;
drop 
  table historic;
drop 
  table rolling;


