CREATE TABLE ami_blink (

    datetime timestamp without time zone NOT NULL,
    element_id varchar(32),
    point_description varchar(32),
    point_name varchar(32),
    message varchar(64),
    value varchar(32)
);

CREATE INDEX idx_ami_blink ON ami_blink(element_id);

SELECT create_hypertable('ami_blink', 'datetime');
SELECT set_chunk_time_interval('ami_blink', INTERVAL '15 Days');
SELECT add_dimension('ami_blink', 'element_id', number_partitions => 82);
