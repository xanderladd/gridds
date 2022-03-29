CREATE TABLE scada_event (

    datetime timestamp without time zone NOT NULL,
    element_id varchar(32),
    point_description varchar(32),
    point_name varchar(32),
    message varchar(64),
    value varchar(32)
);

CREATE INDEX idx_scada_event ON scada_event(element_id);

SELECT create_hypertable('scada_event', 'datetime');
SELECT set_chunk_time_interval('scada_event', INTERVAL '15 Days');
SELECT add_dimension('scada_event', 'element_id', number_partitions => 82);
