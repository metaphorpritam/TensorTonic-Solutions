-- Write your SQL query here
WITH daily_orders AS (
    SELECT 
        order_date
        , COUNT(*) AS daily_count
        , SUM(amount) AS daily_revenue
    FROM orders
    GROUP BY order_date 
    ORDER BY order_date ASC
)
select
    ROUND(AVG(daily_count), 2) AS avg_daily_orders
    , ROUND(AVG(daily_revenue), 2) AS avg_daily_revenue
    , MAX(daily_count) AS busiest_day_orders
from 
    daily_orders;