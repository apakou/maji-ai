from datetime import datetime

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Owner(Base):
    __tablename__ = "owners"
    id = Column(Integer, primary_key=True)
    phone_number = Column(
        String, unique=True, nullable=False
    )  # The primary ID in WhatsApp
    name = Column(String)
    shop_name = Column(String)
    location = Column(String)  # e.g., "Circle Market"
    language_pref = Column(String, default="en")  # English or Twi
    category = Column(String)  # e.g. "Clothing & Footwear"
    record_strength = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class InventoryLog(Base):
    __tablename__ = "inventory_log"
    id = Column(Integer, primary_key=True)
    owner_id = Column(Integer, ForeignKey("owners.id"), nullable=False)
    entry_type = Column(String)  # 'stock_in', 'sale', 'expense'
    product_name = Column(String)
    product_category = Column(String)  # e.g., "Footwear"
    quantity = Column(Integer)
    unit_cost_pesewas = Column(Integer)  # We use integers to avoid float errors
    unit_price_pesewas = Column(Integer)
    raw_message = Column(Text)  # For AI audit trail
    stock_value_pesewas = Column(Integer)  # qty x unit_cost
    logged_at = Column(DateTime, default=datetime.utcnow)


class SusuGroup(Base):
    __tablename__ = "susu_groups"
    id = Column(Integer, primary_key=True)
    group_name = Column(String, nullable=False)
    leader_phone = Column(String, nullable=False)  # The person who collects premiums
    market_location = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class Policy(Base):
    __tablename__ = "policies"
    id = Column(Integer, primary_key=True)
    owner_id = Column(Integer, ForeignKey("owners.id"), nullable=False)
    susu_group_id = Column(Integer, ForeignKey("susu_groups.id"))
    status = Column(String, default="pending")  # 'active', 'lapsed', 'claimed'
    premium_pesewas = Column(Integer, nullable=False)  # e.g., 12000 (GHS 120)
    payout_cap_pesewas = Column(Integer, nullable=False)  # e.g., GHS 10,000
    cover_start_date = Column(Date)
    last_premium_paid_at = Column(DateTime)


class Claim(Base):
    __tablename__ = "claims"
    id = Column(Integer, primary_key=True)
    policy_id = Column(Integer, ForeignKey("policies.id"), nullable=False)
    event_type = Column(String)  # 'fire', 'flood'
    event_date = Column(Date)
    declared_loss_pesewas = Column(Integer)
    verified_loss_pesewas = Column(Integer)  # Based on Visbl AI records
    status = Column(String, default="initiated")
    initiated_at = Column(DateTime, default=datetime.utcnow)


class InventoryDeclaration(Base):
    __tablename__ = "inventory_declarations"
    id = Column(Integer, primary_key=True)
    owner_id = Column(Integer, ForeignKey("owners.id"), nullable=False)
    total_stock_value_ghs = Column(Numeric(12, 2))  # The "Proof of Stock"
    item_breakdown_json = Column(Text)  # AI summary of categories
    generated_at = Column(DateTime, default=datetime.utcnow)
