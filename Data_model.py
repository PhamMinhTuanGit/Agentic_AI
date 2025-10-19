from pydantic import BaseModel
from typing import List, Optional

class Interface(BaseModel):
    name: str
    ip_address: str
    mask: str

class StaticRoute(BaseModel):
    destination: str
    next_hop: str

class OSPF(BaseModel):
    router_id: str
    area: str
    networks: List[str]

class Neighbor(BaseModel):
    neighbor_ip: str
    remote_as: int
    description: Optional[str] = None

class BGP(BaseModel):
    asn: int
    neighbors: List[Neighbor]

class RouterConfig(BaseModel):
    hostname: str
    interfaces: List[Interface]
    static_routes: Optional[List[StaticRoute]] = []
    ospf: Optional[OSPF] = None
    bgp: Optional[BGP] = None
