CREATE TABLE table_name (

    outage_start timestamp without time zone NOT NULL,
    outage_end timestamp without time zone NOT NULL,
    element_id varchar(32),
    element_name varchar(32),
    duration varchar(32),
    customers_affected varchar(32),
    cause varchar(32),
    cause_code varchar(32),
    map_location varchar(32),
    co_op varchar(32),
    feeder varchar(32)
);


