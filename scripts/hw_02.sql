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
CREATE INDEX index_game_id ON battingaverage(batter);
select 
  * 
from 
  battingaverage;
create table rolling as 
select 
  batter, 
  avg(Batting_average) over (
    Partition by batter, 
    (
      case when ud between DATE_SUB(ud, INTERVAL 100 DAY) 
      and ud then Batting_average end
    )
  ) as batting_average
from 
  battingaverage order by batter;
select 
  * 
from 
  rolling;
drop 
  table battingaverage;
drop 
  table rolling;
