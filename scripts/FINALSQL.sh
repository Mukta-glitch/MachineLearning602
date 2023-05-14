echo "Checking environmental variable"
sleep 10
if ! mysql -h localhost -u root -proot -e 'use baseball;'; then
  echo "baseball database does not exists"
  mysql -h localhost -u root -proot -e "create database baseball;"
  echo "loading database..."
  mysql -u root -proot -h localhost --database=baseball < /app/res/baseball.sql
  else
    echo "Database changed"
    mysql -h localhost -u root -proot baseball < /app/FINALSQL.sql
    mysql -h localhost -u root -proot baseball -e '
     SELECT * FROM joined1; '> /app/plot/result.txt
    echo "features data file stored in result.txt"
  fi
# python script
 python FINALMAIN.py
 echo "python script execution done"
