-- Insurance fraud investigation relational schema.
-- This migration defines normalized insurance records and a pgvector-backed
-- claim vector column. It is source code only; it is not executed by this repo.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    date_of_birth DATE NOT NULL,
    occupation TEXT,
    annual_income NUMERIC(12, 2),
    address TEXT,
    city TEXT,
    postcode TEXT,
    phone_number TEXT,
    email TEXT,
    account_created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS policies (
    policy_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE RESTRICT,
    policy_type TEXT NOT NULL,
    coverage_amount NUMERIC(12, 2) NOT NULL,
    premium_amount NUMERIC(12, 2) NOT NULL,
    deductible NUMERIC(12, 2) NOT NULL,
    policy_start_date DATE NOT NULL,
    policy_end_date DATE NOT NULL,
    policy_status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS vehicles (
    vehicle_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE RESTRICT,
    make TEXT NOT NULL,
    model TEXT NOT NULL,
    manufacture_year INTEGER NOT NULL,
    vehicle_type TEXT NOT NULL,
    estimated_value NUMERIC(12, 2),
    registration_region TEXT
);

CREATE TABLE IF NOT EXISTS repair_shops (
    repair_shop_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    city TEXT,
    postcode TEXT,
    owner_name TEXT,
    bank_account_reference TEXT,
    registration_date DATE
);

CREATE TABLE IF NOT EXISTS incidents (
    incident_id TEXT PRIMARY KEY,
    incident_type TEXT NOT NULL,
    incident_date DATE NOT NULL,
    incident_city TEXT,
    incident_address TEXT,
    weather_condition TEXT,
    police_report_reference TEXT,
    witness_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    policy_id TEXT NOT NULL REFERENCES policies(policy_id) ON DELETE RESTRICT,
    vehicle_id TEXT NOT NULL REFERENCES vehicles(vehicle_id) ON DELETE RESTRICT,
    incident_id TEXT NOT NULL REFERENCES incidents(incident_id) ON DELETE RESTRICT,
    repair_shop_id TEXT REFERENCES repair_shops(repair_shop_id) ON DELETE SET NULL,
    claim_date DATE NOT NULL,
    claim_amount NUMERIC(12, 2) NOT NULL,
    repair_cost NUMERIC(12, 2),
    damage_type TEXT,
    injury_reported BOOLEAN NOT NULL DEFAULT FALSE,
    police_report_available BOOLEAN NOT NULL DEFAULT FALSE,
    claim_description TEXT,
    claim_status TEXT NOT NULL,
    historical_fraud_label BOOLEAN,
    investigation_outcome TEXT,
    claim_embedding VECTOR(128),
    CONSTRAINT claims_amount_non_negative CHECK (claim_amount >= 0),
    CONSTRAINT claims_repair_cost_non_negative CHECK (repair_cost IS NULL OR repair_cost >= 0)
);

CREATE TABLE IF NOT EXISTS claim_participants (
    participant_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    participant_type TEXT NOT NULL,
    full_name TEXT,
    phone_number TEXT,
    email TEXT,
    address TEXT,
    relationship_to_claim TEXT
);

CREATE TABLE IF NOT EXISTS payments (
    payment_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    recipient_type TEXT NOT NULL,
    recipient_id TEXT,
    bank_account_reference TEXT,
    payment_amount NUMERIC(12, 2) NOT NULL,
    payment_date DATE NOT NULL,
    payment_status TEXT NOT NULL,
    CONSTRAINT payments_amount_non_negative CHECK (payment_amount >= 0)
);

-- Entity lookup indexes.
CREATE INDEX IF NOT EXISTS idx_customers_city ON customers(city);
CREATE INDEX IF NOT EXISTS idx_customers_postcode ON customers(postcode);
CREATE INDEX IF NOT EXISTS idx_customers_phone_number ON customers(phone_number);
CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(email);

CREATE INDEX IF NOT EXISTS idx_policies_customer_id ON policies(customer_id);
CREATE INDEX IF NOT EXISTS idx_policies_type_status ON policies(policy_type, policy_status);
CREATE INDEX IF NOT EXISTS idx_policies_start_date ON policies(policy_start_date);

CREATE INDEX IF NOT EXISTS idx_vehicles_customer_id ON vehicles(customer_id);
CREATE INDEX IF NOT EXISTS idx_vehicles_type_region ON vehicles(vehicle_type, registration_region);

CREATE INDEX IF NOT EXISTS idx_repair_shops_city ON repair_shops(city);
CREATE INDEX IF NOT EXISTS idx_repair_shops_postcode ON repair_shops(postcode);
CREATE INDEX IF NOT EXISTS idx_repair_shops_bank_account ON repair_shops(bank_account_reference);

CREATE INDEX IF NOT EXISTS idx_incidents_type_city ON incidents(incident_type, incident_city);
CREATE INDEX IF NOT EXISTS idx_incidents_date ON incidents(incident_date);

-- Investigation query indexes.
CREATE INDEX IF NOT EXISTS idx_claims_policy_id ON claims(policy_id);
CREATE INDEX IF NOT EXISTS idx_claims_vehicle_id ON claims(vehicle_id);
CREATE INDEX IF NOT EXISTS idx_claims_incident_id ON claims(incident_id);
CREATE INDEX IF NOT EXISTS idx_claims_repair_shop_id ON claims(repair_shop_id);
CREATE INDEX IF NOT EXISTS idx_claims_claim_date ON claims(claim_date);
CREATE INDEX IF NOT EXISTS idx_claims_amount ON claims(claim_amount);
CREATE INDEX IF NOT EXISTS idx_claims_status ON claims(claim_status);
CREATE INDEX IF NOT EXISTS idx_claims_historical_fraud_label ON claims(historical_fraud_label);

CREATE INDEX IF NOT EXISTS idx_claim_participants_claim_id ON claim_participants(claim_id);
CREATE INDEX IF NOT EXISTS idx_claim_participants_phone ON claim_participants(phone_number);
CREATE INDEX IF NOT EXISTS idx_claim_participants_email ON claim_participants(email);
CREATE INDEX IF NOT EXISTS idx_claim_participants_address ON claim_participants(address);

CREATE INDEX IF NOT EXISTS idx_payments_claim_id ON payments(claim_id);
CREATE INDEX IF NOT EXISTS idx_payments_bank_account ON payments(bank_account_reference);
CREATE INDEX IF NOT EXISTS idx_payments_recipient ON payments(recipient_type, recipient_id);
CREATE INDEX IF NOT EXISTS idx_payments_date ON payments(payment_date);

-- pgvector cosine index for candidate retrieval. The application still uses SQL
-- after vector retrieval to load complete relational evidence and apply exact filters.
CREATE INDEX IF NOT EXISTS idx_claims_embedding_cosine
    ON claims USING ivfflat (claim_embedding vector_cosine_ops)
    WITH (lists = 100)
    WHERE claim_embedding IS NOT NULL;
