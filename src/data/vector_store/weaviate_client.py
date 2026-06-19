import weaviate
from weaviate.collections.classes.filters import Filter

from config.configuration import get_settings

FACILITY_INFO_COLLECTION = "FacilityInfo"
PARKING_DETAILS_COLLECTION = "ParkingDetails"


_client = None


def connect_weaviate() -> weaviate.WeaviateClient:
    """Open a new Weaviate connection from settings (caller owns/closes it).

    Host, ports, and the OpenAI key used by ``text2vec-openai`` all come from
    the central :class:`Settings` — nothing is hardcoded here.
    """
    settings = get_settings()
    return weaviate.connect_to_local(
        host=settings.weaviate.host,
        port=settings.weaviate.http_port,
        grpc_port=settings.weaviate.grpc_port,
        headers={"X-OpenAI-Api-Key": settings.openai.api_key.get_secret_value()},
    )


def get_weaviate_client() -> weaviate.WeaviateClient:
    global _client
    if _client is None or not _client.is_connected():
        _client = connect_weaviate()
    return _client


def search_facility_info(query: str, category: str | None = None, k: int = 3):
    with get_weaviate_client() as client:
        collection = client.collections.get(FACILITY_INFO_COLLECTION)
        filters = Filter.by_property("category").equal(category) if category else None
        response = collection.query.hybrid(
            query=query,
            alpha=0.5,
            filters=filters,
            limit=k,
        )
    return response

def search_parking_details(query: str, floor: int | None = None, zone_name: str | None = None, k: int = 3):
    with get_weaviate_client() as client:
        collection = client.collections.get(PARKING_DETAILS_COLLECTION)
        filter_list = []
        if floor is not None:
            filter_list.append(Filter.by_property("floor").equal(floor))
        if zone_name is not None:
            filter_list.append(Filter.by_property("zone_name").equal(zone_name))
        filters = Filter.all_of(filter_list) if filter_list else None
        response = collection.query.hybrid(
            query=query,
            alpha=0.5,
            filters=filters,
            limit=k,
        )
    return response