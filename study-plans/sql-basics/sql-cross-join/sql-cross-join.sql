-- Write your SQL query here
select
    s.segment_name
    , m.metric_name
FROM segments AS s CROSS JOIN
    metrics AS m 
ORDER BY s.segment_name ASC, m.metric_name ASC;