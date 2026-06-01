-- Write your SQL query here
SELECT
    c.name
    , c.city 
    , SUM(COALESCE(o.amount,0)) AS total_spent
FROM customers AS c 
    LEFT JOIN orders AS o 
    ON c.id = o.customer_id
GROUP BY c.name, c.city
ORDER BY total_spent DESC, name ASC;