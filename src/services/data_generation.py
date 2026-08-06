"""Synthetic insurance data generation for investigation workflows.

The generator creates deterministic, linked relational records that exercise exact
SQL retrieval and later semantic similarity workflows. Claim embeddings are left
empty because vectors are produced by the custom embedding pipeline.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

TABLE_WRITE_ORDER = (
    "customers",
    "policies",
    "vehicles",
    "repair_shops",
    "incidents",
    "claims",
    "claim_participants",
    "payments",
)


@dataclass(frozen=True)
class SyntheticDataConfig:
    """Configuration for deterministic synthetic insurance data generation."""

    seed: int = 2410
    customer_count: int = 200
    repair_shop_count: int = 30
    normal_claim_count: int = 500
    fraudulent_claim_count: int = 80
    coordinated_group_count: int = 6
    coordinated_group_size: int = 8
    output_dir: Path = Path("data/generated")
    start_date: date = date(2023, 1, 1)
    end_date: date = date(2025, 12, 31)

    @property
    def claim_count(self) -> int:
        """Return the total generated claim count."""
        return (
            self.normal_claim_count
            + self.fraudulent_claim_count
            + self.coordinated_group_count * self.coordinated_group_size
        )


class SyntheticInsuranceDataGenerator:
    """Generate linked synthetic insurance records with investigative patterns."""

    first_names = (
        "Aarav",
        "Maya",
        "Noah",
        "Sophia",
        "Liam",
        "Olivia",
        "Ethan",
        "Isla",
        "Arjun",
        "Amelia",
        "Rohan",
        "Freya",
    )
    last_names = (
        "Shah",
        "Patel",
        "Mehta",
        "Kapoor",
        "Williams",
        "Brown",
        "Taylor",
        "Walker",
        "Khan",
        "Singh",
        "Wilson",
        "Evans",
    )
    cities = ("London", "Manchester", "Birmingham", "Leeds", "Bristol", "Glasgow")
    occupations = (
        "Accountant",
        "Driver",
        "Engineer",
        "Nurse",
        "Teacher",
        "Contractor",
        "Mechanic",
        "Consultant",
    )
    vehicle_makes = ("Ford", "Toyota", "BMW", "Nissan", "Audi", "Honda", "Tesla")
    vehicle_models = ("Focus", "Corolla", "X3", "Qashqai", "A4", "Civic", "Model 3")
    vehicle_types = ("sedan", "suv", "hatchback", "van", "motorcycle")
    incident_types = ("collision", "vehicle_theft", "fire", "flood", "vandalism")
    damage_types = ("front_end", "rear_end", "side_panel", "total_loss", "water_damage")
    weather_conditions = ("clear", "rain", "fog", "snow", "wind")
    policy_types = ("comprehensive", "third_party", "collision", "theft")

    def __init__(self, config: SyntheticDataConfig) -> None:
        self.config = config
        self.random = random.Random(config.seed)

    def generate(self) -> dict[str, list[dict[str, Any]]]:
        """Generate a complete linked dataset in table-load order."""
        customers = self._generate_customers()
        policies = self._generate_policies(customers)
        vehicles = self._generate_vehicles(customers)
        repair_shops = self._generate_repair_shops()
        incidents: list[dict[str, Any]] = []
        claims: list[dict[str, Any]] = []
        participants: list[dict[str, Any]] = []
        payments: list[dict[str, Any]] = []

        claim_index = 1
        for _ in range(self.config.normal_claim_count):
            self._append_claim_bundle(
                claim_index=claim_index,
                pattern="normal",
                customers=customers,
                policies=policies,
                vehicles=vehicles,
                repair_shops=repair_shops,
                incidents=incidents,
                claims=claims,
                participants=participants,
                payments=payments,
            )
            claim_index += 1

        for _ in range(self.config.fraudulent_claim_count):
            self._append_claim_bundle(
                claim_index=claim_index,
                pattern="historical_fraud",
                customers=customers,
                policies=policies,
                vehicles=vehicles,
                repair_shops=repair_shops,
                incidents=incidents,
                claims=claims,
                participants=participants,
                payments=payments,
            )
            claim_index += 1

        for group_index in range(1, self.config.coordinated_group_count + 1):
            group_context = self._build_coordinated_group_context(group_index, repair_shops)
            for member_index in range(self.config.coordinated_group_size):
                self._append_claim_bundle(
                    claim_index=claim_index,
                    pattern="coordinated_group",
                    customers=customers,
                    policies=policies,
                    vehicles=vehicles,
                    repair_shops=repair_shops,
                    incidents=incidents,
                    claims=claims,
                    participants=participants,
                    payments=payments,
                    group_context=group_context,
                    force_historical_fraud=(member_index % 3 != 0),
                )
                claim_index += 1

        return {
            "customers": customers,
            "policies": policies,
            "vehicles": vehicles,
            "repair_shops": repair_shops,
            "incidents": incidents,
            "claims": claims,
            "claim_participants": participants,
            "payments": payments,
        }

    def write(self, dataset: dict[str, list[dict[str, Any]]] | None = None) -> None:
        """Write generated records as JSON files under the configured output directory."""
        dataset = dataset or self.generate()
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        for table_name in TABLE_WRITE_ORDER:
            path = output_dir / f"{table_name}.json"
            path.write_text(
                json.dumps(dataset[table_name], indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        manifest = {
            "seed": self.config.seed,
            "tables": {table_name: len(dataset[table_name]) for table_name in TABLE_WRITE_ORDER},
            "claim_embedding_policy": "claim embeddings are produced by the embedding pipeline",
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _generate_customers(self) -> list[dict[str, Any]]:
        customers = []
        shared_addresses = [
            "14 Northgate Road",
            "88 Market Street",
            "5 Bridge Lane",
        ]
        shared_phones = ["+447700900101", "+447700900202", "+447700900303"]

        for index in range(1, self.config.customer_count + 1):
            first_name = self.random.choice(self.first_names)
            last_name = self.random.choice(self.last_names)
            city = self.random.choice(self.cities)
            use_shared_address = index % 17 == 0
            use_shared_phone = index % 19 == 0
            address = (
                self.random.choice(shared_addresses)
                if use_shared_address
                else f"{self.random.randint(1, 220)} {self.random.choice(['Oak', 'King', 'Station', 'Mill'])} Road"
            )
            phone = (
                self.random.choice(shared_phones)
                if use_shared_phone
                else f"+447700{900000 + index:06d}"
            )
            customers.append(
                {
                    "customer_id": f"CUS-{index:05d}",
                    "full_name": f"{first_name} {last_name}",
                    "date_of_birth": self._random_date(date(1960, 1, 1), date(2002, 12, 31)),
                    "occupation": self.random.choice(self.occupations),
                    "annual_income": self.random.randint(24000, 130000),
                    "address": address,
                    "city": city,
                    "postcode": self._postcode(city, index),
                    "phone_number": phone,
                    "email": f"{first_name.lower()}.{last_name.lower()}{index}@example.com",
                    "account_created_at": f"{self._random_date(date(2020, 1, 1), date(2024, 12, 31))}T09:00:00Z",
                }
            )
        return customers

    def _generate_policies(self, customers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        policies = []
        for index, customer in enumerate(customers, start=1):
            start = self._random_date(self.config.start_date, date(2025, 6, 30))
            policies.append(
                {
                    "policy_id": f"POL-{index:05d}",
                    "customer_id": customer["customer_id"],
                    "policy_type": self.random.choice(self.policy_types),
                    "coverage_amount": self.random.choice([15000, 25000, 50000, 75000, 100000]),
                    "premium_amount": self.random.randint(450, 2600),
                    "deductible": self.random.choice([250, 500, 750, 1000]),
                    "policy_start_date": start,
                    "policy_end_date": start + timedelta(days=365),
                    "policy_status": self.random.choice(["active", "active", "active", "expired"]),
                }
            )
        return policies

    def _generate_vehicles(self, customers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        vehicles = []
        for index, customer in enumerate(customers, start=1):
            make = self.random.choice(self.vehicle_makes)
            model = self.random.choice(self.vehicle_models)
            vehicles.append(
                {
                    "vehicle_id": f"VEH-{index:05d}",
                    "customer_id": customer["customer_id"],
                    "make": make,
                    "model": model,
                    "manufacture_year": self.random.randint(2012, 2025),
                    "vehicle_type": self.random.choice(self.vehicle_types),
                    "estimated_value": self.random.randint(6000, 70000),
                    "registration_region": self.random.choice(self.cities),
                }
            )
        return vehicles

    def _generate_repair_shops(self) -> list[dict[str, Any]]:
        shops = []
        shared_bank_accounts = ["BANK-SHARED-001", "BANK-SHARED-002", "BANK-SHARED-003"]
        for index in range(1, self.config.repair_shop_count + 1):
            city = self.random.choice(self.cities)
            bank_account = (
                self.random.choice(shared_bank_accounts)
                if index % 9 == 0
                else f"BANK-RS-{index:05d}"
            )
            shops.append(
                {
                    "repair_shop_id": f"RS-{index:04d}",
                    "name": f"{self.random.choice(['Premier', 'Metro', 'Crown', 'Union'])} Auto Repair {index}",
                    "city": city,
                    "postcode": self._postcode(city, index),
                    "owner_name": f"{self.random.choice(self.first_names)} {self.random.choice(self.last_names)}",
                    "bank_account_reference": bank_account,
                    "registration_date": self._random_date(date(2016, 1, 1), date(2025, 1, 1)),
                }
            )
        return shops

    def _append_claim_bundle(
        self,
        *,
        claim_index: int,
        pattern: str,
        customers: list[dict[str, Any]],
        policies: list[dict[str, Any]],
        vehicles: list[dict[str, Any]],
        repair_shops: list[dict[str, Any]],
        incidents: list[dict[str, Any]],
        claims: list[dict[str, Any]],
        participants: list[dict[str, Any]],
        payments: list[dict[str, Any]],
        group_context: dict[str, Any] | None = None,
        force_historical_fraud: bool | None = None,
    ) -> None:
        customer_position = self.random.randrange(len(customers))
        customer = customers[customer_position]
        policy = policies[customer_position]
        vehicle = vehicles[customer_position]
        shop = self._select_repair_shop(pattern, repair_shops, group_context)

        policy_start = date.fromisoformat(str(policy["policy_start_date"]))
        claim_date = self._claim_date_for_pattern(pattern, policy_start)
        incident_date = claim_date - timedelta(days=self.random.randint(0, 14))
        claim_id = f"CLM-{claim_index:05d}"
        incident_id = f"INC-{claim_index:05d}"
        fraud_label = force_historical_fraud
        if fraud_label is None:
            fraud_label = pattern in {"historical_fraud", "coordinated_group"}

        incident_city = group_context.get("city") if group_context else self.random.choice(self.cities)
        incident_type = (
            group_context.get("incident_type")
            if group_context
            else self.random.choice(self.incident_types)
        )
        claim_amount = self._claim_amount_for_pattern(pattern)
        repair_cost = max(500, int(claim_amount * self.random.uniform(0.45, 0.95)))

        incidents.append(
            {
                "incident_id": incident_id,
                "incident_type": incident_type,
                "incident_date": incident_date,
                "incident_city": incident_city,
                "incident_address": group_context.get("incident_address")
                if group_context
                else f"{self.random.randint(1, 250)} {self.random.choice(['High', 'Canal', 'Park'])} Street",
                "weather_condition": self.random.choice(self.weather_conditions),
                "police_report_reference": f"POLICE-{claim_index:06d}"
                if self.random.random() > 0.35
                else None,
                "witness_count": self.random.randint(0, 4),
            }
        )

        claims.append(
            {
                "claim_id": claim_id,
                "policy_id": policy["policy_id"],
                "vehicle_id": vehicle["vehicle_id"],
                "incident_id": incident_id,
                "repair_shop_id": shop["repair_shop_id"],
                "claim_date": claim_date,
                "claim_amount": claim_amount,
                "repair_cost": repair_cost,
                "damage_type": group_context.get("damage_type")
                if group_context
                else self.random.choice(self.damage_types),
                "injury_reported": self.random.random() < (0.25 if pattern == "normal" else 0.45),
                "police_report_available": self.random.random() > (0.25 if pattern == "normal" else 0.55),
                "claim_description": self._claim_description(pattern, incident_type, incident_city),
                "claim_status": self.random.choice(["open", "under_review", "paid", "closed"]),
                "historical_fraud_label": fraud_label,
                "investigation_outcome": "confirmed_fraud" if fraud_label else "not_fraud",
                "claim_embedding": None,
            }
        )

        self._append_participants(claim_id, customer, participants, group_context)
        self._append_payments(claim_id, shop, claim_amount, payments, group_context)

    def _build_coordinated_group_context(
        self,
        group_index: int,
        repair_shops: list[dict[str, Any]],
    ) -> dict[str, Any]:
        shop = repair_shops[group_index % len(repair_shops)]
        city = self.random.choice(self.cities)
        return {
            "group_id": f"GRP-{group_index:03d}",
            "repair_shop_id": shop["repair_shop_id"],
            "shared_bank_account": f"BANK-GROUP-{group_index:03d}",
            "shared_phone_number": f"+447701{group_index:06d}",
            "shared_address": f"{20 + group_index} Warehouse Lane",
            "incident_type": self.random.choice(["vehicle_theft", "collision"]),
            "incident_address": f"Unit {group_index}, Dockside Estate",
            "damage_type": self.random.choice(["total_loss", "front_end"]),
            "city": city,
        }

    def _select_repair_shop(
        self,
        pattern: str,
        repair_shops: list[dict[str, Any]],
        group_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if group_context:
            repair_shop_id = group_context["repair_shop_id"]
            return next(shop for shop in repair_shops if shop["repair_shop_id"] == repair_shop_id)
        if pattern == "historical_fraud":
            suspicious_shops = repair_shops[:: max(1, len(repair_shops) // 5)]
            return self.random.choice(suspicious_shops)
        return self.random.choice(repair_shops)

    def _claim_date_for_pattern(self, pattern: str, policy_start: date) -> date:
        if pattern in {"historical_fraud", "coordinated_group"}:
            return policy_start + timedelta(days=self.random.randint(3, 45))
        latest_offset = max(90, (self.config.end_date - policy_start).days)
        return policy_start + timedelta(days=self.random.randint(60, latest_offset))

    def _claim_amount_for_pattern(self, pattern: str) -> int:
        if pattern == "normal":
            return self.random.randint(1200, 22000)
        if pattern == "coordinated_group":
            return self.random.randint(15000, 55000)
        return self.random.randint(18000, 75000)

    def _append_participants(
        self,
        claim_id: str,
        customer: dict[str, Any],
        participants: list[dict[str, Any]],
        group_context: dict[str, Any] | None,
    ) -> None:
        participant_index = len(participants) + 1
        participants.append(
            {
                "participant_id": f"PAR-{participant_index:06d}",
                "claim_id": claim_id,
                "participant_type": "claimant",
                "full_name": customer["full_name"],
                "phone_number": customer["phone_number"],
                "email": customer["email"],
                "address": customer["address"],
                "relationship_to_claim": "policy_holder",
            }
        )
        if self.random.random() < 0.65 or group_context:
            participant_index += 1
            participants.append(
                {
                    "participant_id": f"PAR-{participant_index:06d}",
                    "claim_id": claim_id,
                    "participant_type": "third_party",
                    "full_name": f"{self.random.choice(self.first_names)} {self.random.choice(self.last_names)}",
                    "phone_number": group_context.get("shared_phone_number")
                    if group_context
                    else f"+447702{participant_index:06d}",
                    "email": f"participant{participant_index}@example.com",
                    "address": group_context.get("shared_address")
                    if group_context
                    else f"{self.random.randint(1, 300)} West Road",
                    "relationship_to_claim": "other_driver",
                }
            )

    def _append_payments(
        self,
        claim_id: str,
        shop: dict[str, Any],
        claim_amount: int,
        payments: list[dict[str, Any]],
        group_context: dict[str, Any] | None,
    ) -> None:
        payment_index = len(payments) + 1
        payment_count = 2 if claim_amount > 30000 or group_context else 1
        remaining = claim_amount
        for item in range(payment_count):
            amount = remaining if item == payment_count - 1 else int(claim_amount * 0.65)
            remaining -= amount
            payments.append(
                {
                    "payment_id": f"PAY-{payment_index:06d}",
                    "claim_id": claim_id,
                    "recipient_type": "repair_shop" if item == 0 else "claimant",
                    "recipient_id": shop["repair_shop_id"] if item == 0 else None,
                    "bank_account_reference": group_context.get("shared_bank_account")
                    if group_context
                    else shop["bank_account_reference"],
                    "payment_amount": amount,
                    "payment_date": self._random_date(date(2023, 1, 1), date(2025, 12, 31)),
                    "payment_status": self.random.choice(["paid", "paid", "pending", "review"]),
                }
            )
            payment_index += 1

    def _claim_description(self, pattern: str, incident_type: str, city: str) -> str:
        if pattern == "normal":
            return f"Customer reported {incident_type} incident in {city}."
        if pattern == "coordinated_group":
            return f"Claim shares coordinated-group repair and payment attributes for {incident_type}."
        return f"Historically flagged {incident_type} claim with elevated amount and linked entities."

    def _random_date(self, start: date, end: date) -> date:
        days = (end - start).days
        return start + timedelta(days=self.random.randint(0, max(days, 0)))

    def _postcode(self, city: str, index: int) -> str:
        prefix = {
            "London": "LDN",
            "Manchester": "MAN",
            "Birmingham": "BIR",
            "Leeds": "LDS",
            "Bristol": "BST",
            "Glasgow": "GLA",
        }[city]
        return f"{prefix} {index % 90:02d}AA"


def create_sample_dataset() -> dict[str, list[dict[str, Any]]]:
    """Return a tiny linked fixture that demonstrates the generated shape."""
    return {
        "customers": [
            {
                "customer_id": "CUS-00001",
                "full_name": "Maya Patel",
                "date_of_birth": "1987-04-12",
                "occupation": "Consultant",
                "annual_income": 72000,
                "address": "14 Northgate Road",
                "city": "London",
                "postcode": "LDN 01AA",
                "phone_number": "+447700900101",
                "email": "maya.patel@example.com",
                "account_created_at": "2023-02-01T09:00:00Z",
            },
            {
                "customer_id": "CUS-00002",
                "full_name": "Rohan Shah",
                "date_of_birth": "1979-09-03",
                "occupation": "Driver",
                "annual_income": 41000,
                "address": "22 Market Street",
                "city": "London",
                "postcode": "LDN 02AA",
                "phone_number": "+447700900202",
                "email": "rohan.shah@example.com",
                "account_created_at": "2023-03-15T09:00:00Z",
            },
        ],
        "policies": [
            {
                "policy_id": "POL-00001",
                "customer_id": "CUS-00001",
                "policy_type": "comprehensive",
                "coverage_amount": 75000,
                "premium_amount": 1800,
                "deductible": 500,
                "policy_start_date": "2025-01-01",
                "policy_end_date": "2026-01-01",
                "policy_status": "active",
            },
            {
                "policy_id": "POL-00002",
                "customer_id": "CUS-00002",
                "policy_type": "theft",
                "coverage_amount": 50000,
                "premium_amount": 1200,
                "deductible": 750,
                "policy_start_date": "2025-01-20",
                "policy_end_date": "2026-01-20",
                "policy_status": "active",
            },
        ],
        "vehicles": [
            {
                "vehicle_id": "VEH-00001",
                "customer_id": "CUS-00001",
                "make": "BMW",
                "model": "X3",
                "manufacture_year": 2021,
                "vehicle_type": "suv",
                "estimated_value": 42000,
                "registration_region": "London",
            },
            {
                "vehicle_id": "VEH-00002",
                "customer_id": "CUS-00002",
                "make": "Audi",
                "model": "A4",
                "manufacture_year": 2020,
                "vehicle_type": "sedan",
                "estimated_value": 31000,
                "registration_region": "London",
            },
        ],
        "repair_shops": [
            {
                "repair_shop_id": "RS-0001",
                "name": "Premier Auto Repair 1",
                "city": "London",
                "postcode": "LDN 11AA",
                "owner_name": "Noah Singh",
                "bank_account_reference": "BANK-SHARED-001",
                "registration_date": "2020-06-01",
            },
            {
                "repair_shop_id": "RS-0002",
                "name": "Metro Auto Repair 2",
                "city": "London",
                "postcode": "LDN 12AA",
                "owner_name": "Sophia Brown",
                "bank_account_reference": "BANK-RS-00002",
                "registration_date": "2019-04-18",
            },
        ],
        "incidents": [
            {
                "incident_id": "INC-00001",
                "incident_type": "vehicle_theft",
                "incident_date": "2025-01-29",
                "incident_city": "London",
                "incident_address": "Unit 1, Dockside Estate",
                "weather_condition": "rain",
                "police_report_reference": "POLICE-000001",
                "witness_count": 1,
            },
            {
                "incident_id": "INC-00002",
                "incident_type": "collision",
                "incident_date": "2025-05-11",
                "incident_city": "London",
                "incident_address": "45 High Street",
                "weather_condition": "clear",
                "police_report_reference": "POLICE-000002",
                "witness_count": 2,
            },
        ],
        "claims": [
            {
                "claim_id": "CLM-00001",
                "policy_id": "POL-00001",
                "vehicle_id": "VEH-00001",
                "incident_id": "INC-00001",
                "repair_shop_id": "RS-0001",
                "claim_date": "2025-02-03",
                "claim_amount": 28000,
                "repair_cost": 21000,
                "damage_type": "total_loss",
                "injury_reported": False,
                "police_report_available": True,
                "claim_description": "Vehicle-theft claim filed shortly after policy creation.",
                "claim_status": "under_review",
                "historical_fraud_label": True,
                "investigation_outcome": "confirmed_fraud",
                "claim_embedding": None,
            },
            {
                "claim_id": "CLM-00002",
                "policy_id": "POL-00002",
                "vehicle_id": "VEH-00002",
                "incident_id": "INC-00002",
                "repair_shop_id": "RS-0001",
                "claim_date": "2025-05-14",
                "claim_amount": 16500,
                "repair_cost": 12000,
                "damage_type": "front_end",
                "injury_reported": True,
                "police_report_available": True,
                "claim_description": "Collision claim linked to a repair shop used by another claim.",
                "claim_status": "paid",
                "historical_fraud_label": False,
                "investigation_outcome": "not_fraud",
                "claim_embedding": None,
            },
        ],
        "claim_participants": [
            {
                "participant_id": "PAR-000001",
                "claim_id": "CLM-00001",
                "participant_type": "claimant",
                "full_name": "Maya Patel",
                "phone_number": "+447700900101",
                "email": "maya.patel@example.com",
                "address": "14 Northgate Road",
                "relationship_to_claim": "policy_holder",
            },
            {
                "participant_id": "PAR-000002",
                "claim_id": "CLM-00002",
                "participant_type": "claimant",
                "full_name": "Rohan Shah",
                "phone_number": "+447700900202",
                "email": "rohan.shah@example.com",
                "address": "22 Market Street",
                "relationship_to_claim": "policy_holder",
            },
        ],
        "payments": [
            {
                "payment_id": "PAY-000001",
                "claim_id": "CLM-00001",
                "recipient_type": "repair_shop",
                "recipient_id": "RS-0001",
                "bank_account_reference": "BANK-SHARED-001",
                "payment_amount": 21000,
                "payment_date": "2025-02-14",
                "payment_status": "paid",
            },
            {
                "payment_id": "PAY-000002",
                "claim_id": "CLM-00002",
                "recipient_type": "repair_shop",
                "recipient_id": "RS-0001",
                "bank_account_reference": "BANK-SHARED-001",
                "payment_amount": 12000,
                "payment_date": "2025-05-20",
                "payment_status": "paid",
            },
        ],
    }
