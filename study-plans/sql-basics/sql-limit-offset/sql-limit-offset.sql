-- Write your SQL query here
select
    product
    , revenue
    , sale_date
FROM sales
ORDER BY revenue desc, sale_date ASC
OFFSET 1
LIMIT 3;