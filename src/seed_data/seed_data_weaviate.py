import json

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.collections import Collection

FACILITY_INFO_COLLECTION = "FacilityInfo"
PARKING_DETAILS_COLLECTION = "ParkingDetails"


def seed_weaviate_database():
    client = weaviate.connect_to_local(headers={
        "X-OpenAI-Api-Key": "" # Replace with your key
    })
    try:
        existing = {c.name for c in client.collections.list_all().values()}
        if FACILITY_INFO_COLLECTION not in existing:
            client.collections.create(
                name=FACILITY_INFO_COLLECTION,
                properties=[
                    Property(name="description", data_type=DataType.TEXT),
                    Property(name="category", data_type=DataType.TEXT, skip_vectorization=True),
                    Property(name="source_doc", data_type=DataType.TEXT, skip_vectorization=True),
                ],
                vectorizer_config=Configure.Vectorizer.text2vec_openai(),
            )

        if PARKING_DETAILS_COLLECTION not in existing:
            client.collections.create(
                name=PARKING_DETAILS_COLLECTION,
                properties=[
                    Property(name="zone_name", data_type=DataType.TEXT, skip_vectorization=True),
                    Property(name="floor", data_type=DataType.INT, skip_vectorization=True),
                    Property(name="capacity_total", data_type=DataType.INT, skip_vectorization=True),
                    Property(name="amenities", data_type=DataType.TEXT_ARRAY, skip_vectorization=True),
                    Property(name="description", data_type=DataType.TEXT),
                ],
                vectorizer_config=Configure.Vectorizer.text2vec_openai(),
            )

        facility_info_json_data = read_json_data("weaviate_facility_info.json")
        facility_info_collection = client.collections.get(FACILITY_INFO_COLLECTION)
        write_into_weaviate_collection(facility_info_collection, facility_info_json_data)
        parking_details_collection = client.collections.get(PARKING_DETAILS_COLLECTION)
        parking_details_json_data = read_json_data("weaviate_parking_details.json")
        write_into_weaviate_collection(parking_details_collection, parking_details_json_data)
    finally:
        client.close()


def write_into_weaviate_collection(weaviate_collection: Collection, json_data: dict):
    with weaviate_collection.batch.dynamic() as batch:
        for json_obj in json_data:
            batch.add_object(json_obj)

    if weaviate_collection.batch.failed_objects:
        print(f"Failed to import {len(weaviate_collection.batch.failed_objects)} objects.")
        for failed in weaviate_collection.batch.failed_objects:
            print(f"Error: {failed.message}")


def read_json_data(file_path: str) -> dict:
    with open(file_path) as json_file:
        return json.load(json_file)


def read_weaviate_collection(collection_name: str):
    client = weaviate.connect_to_local()
    collection = client.collections.get(collection_name)
    response = collection.aggregate.over_all(total_count=True)

    print(f"Total objects in collection: {response.total_count}")
    # Read all objects
    try:
        for item in collection.iterator():
            print(item.uuid)
            print(item.properties)
    finally:
        client.close()

def clean_up_weaviate_collection():
    client = weaviate.connect_to_local()
    try:
        # 1. Get a list of all collection names
        collections = client.collections.list_all()

        if not collections:
            print("No collections found to delete.")
        else:
            # 2. Iterate and delete
            for name in collections:
                print(f"Deleting collection: {name}")
                client.collections.delete(name)

            print("\nAll collections have been dropped successfully.")

    finally:
        client.close()

if __name__ == '__main__':
    print("Drop collections if exists.")
    clean_up_weaviate_collection()
    print("Seed weaviate database")
    seed_weaviate_database()
    print("Read test - FACILITY_INFO_COLLECTION--------------------------")
    read_weaviate_collection(FACILITY_INFO_COLLECTION)
    print("Read test - PARKING_DETAILS_COLLECTION--------------------------")
    read_weaviate_collection(PARKING_DETAILS_COLLECTION)
