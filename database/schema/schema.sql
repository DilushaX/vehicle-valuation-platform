CREATE TABLE IF NOT EXISTS vehicles (
    id SERIAL PRIMARY KEY,
    category VARCHAR(50),
    brand VARCHAR(100),
    model VARCHAR(150),
    manufacture_year INTEGER,
    registration_year INTEGER,
    fuel_type VARCHAR(50),
    transmission VARCHAR(50),
    engine_cc INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS listings (
    id SERIAL PRIMARY KEY,
    listing_id VARCHAR(100) NOT NULL UNIQUE,
    vehicle_id INTEGER NOT NULL REFERENCES vehicles(id),
    listing_url TEXT NOT NULL,
    title TEXT,
    description TEXT,
    location VARCHAR(150),
    district VARCHAR(100),
    source VARCHAR(50) NOT NULL DEFAULT 'riyasewana',
    current_status VARCHAR(50) NOT NULL DEFAULT 'ACTIVE',
    first_seen_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    validation_issues TEXT,
    quality_score NUMERIC(5,2),
    ml_eligible BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS price_history (
    id SERIAL PRIMARY KEY,
    listing_id INTEGER NOT NULL REFERENCES listings(id),
    price INTEGER,
    observed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS scrape_runs (
    id SERIAL PRIMARY KEY,
    source VARCHAR(50) NOT NULL DEFAULT 'riyasewana',
    category VARCHAR(100),
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    pages_requested INTEGER NOT NULL DEFAULT 0,
    pages_scraped INTEGER NOT NULL DEFAULT 0,
    failed_pages INTEGER NOT NULL DEFAULT 0,
    listings_found INTEGER NOT NULL DEFAULT 0,
    new_listings INTEGER NOT NULL DEFAULT 0,
    updated_listings INTEGER NOT NULL DEFAULT 0,
    errors TEXT,
    status VARCHAR(30) NOT NULL DEFAULT 'RUNNING'
);

CREATE TABLE IF NOT EXISTS listing_observations (
    id SERIAL PRIMARY KEY,
    listing_id INTEGER NOT NULL REFERENCES listings(id),
    scrape_run_id INTEGER NOT NULL REFERENCES scrape_runs(id),
    observed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    observed_price INTEGER,
    observed_mileage INTEGER,
    availability VARCHAR(30) NOT NULL DEFAULT 'AVAILABLE'
);

CREATE INDEX IF NOT EXISTS idx_listing_status
ON listings(current_status);

CREATE INDEX IF NOT EXISTS idx_listing_source
ON listings(source);

CREATE INDEX IF NOT EXISTS idx_price_history_listing
ON price_history(listing_id);

CREATE INDEX IF NOT EXISTS idx_price_history_date
ON price_history(observed_at);

CREATE INDEX IF NOT EXISTS idx_observation_listing
ON listing_observations(listing_id);

CREATE INDEX IF NOT EXISTS idx_observation_scrape
ON listing_observations(scrape_run_id);

CREATE INDEX IF NOT EXISTS idx_observation_date
ON listing_observations(observed_at);