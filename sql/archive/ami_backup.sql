CREATE TABLE ami_grafana (

    start_time timestamp without time zone NOT NULL,
    end_time timestamp without time zone NOT NULL,
    element_id varchar(32),
    feeder varchar(32),
    substation varchar(32),
    map_location varchar(32),
    -- kwh varchar(32),
    kwh float(32),
    co_op varchar(32)
);

-- CREATE INDEX idx_ami ON ami_grafana(element_id);

SELECT create_hypertable('ami_grafana', 'start_time');
SELECT set_chunk_time_interval('ami_grafana', INTERVAL '15 Days');
SELECT add_dimension('ami_grafana', 'element_id', number_partitions => 82);


      