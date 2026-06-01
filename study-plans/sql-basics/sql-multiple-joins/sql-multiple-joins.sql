-- Write your SQL query here
select
    u.username
    , ea.experiment_name
    , ea.variant
    , c.revenue
FROM users AS u
    INNER JOIN experiment_assignments AS ea
        ON u.id = ea.user_id
    INNER JOIN conversions AS c
        ON u.id = c.user_id
ORDER BY ea.experiment_name ASC, revenue DESC, username ASC;
    