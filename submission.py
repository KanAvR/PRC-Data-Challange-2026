import argparse
import json
from minio import Minio

BUCKET = "prc-2026-humble-giraffe"
TEAM = "humble-giraffe"


def get_client():
    with open("credentials.json") as f:
        creds = json.load(f)
    return Minio(
        "s3.opensky-network.org",
        access_key=creds["accessKey"],
        secret_key=creds["secretKey"],
        secure=True,
    )


def remote_name(version):
    version = version if version.startswith("v") else f"v{version}"
    return f"{TEAM}_{version}.parquet"


def cmd_list(client, args):
    try:
        for obj in client.list_objects(BUCKET, recursive=True):
            print(obj.object_name)
    except Exception as e:
        print("Error listing bucket:", e)


def cmd_read(client, args):
    name = remote_name(args.version) + "_result.json"
    try:
        response = client.get_object(BUCKET, name)
        print(response.read().decode())
        response.close()
        response.release_conn()
    except Exception as e:
        print("Result not available yet:", e)


def cmd_upload(client, args):
    remote_file = remote_name(args.version)
    client.fput_object(BUCKET, remote_file, args.file)
    print(f"Uploaded: s3://{BUCKET}/{remote_file}")


def main():
    parser = argparse.ArgumentParser(description="Manage PRC 2026 submissions.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List objects in the bucket").set_defaults(func=cmd_list)

    p_read = sub.add_parser("read", help="Print the scoring result for a version")
    p_read.add_argument("version", help="Submission version, e.g. 6 or v6")
    p_read.set_defaults(func=cmd_read)

    p_upload = sub.add_parser("upload", help="Upload a submission file as a version")
    p_upload.add_argument("version", help="Submission version, e.g. 6 or v6")
    p_upload.add_argument(
        "--file",
        default="output/submitting.parquet",
        help="Local parquet file to upload (default: %(default)s)",
    )
    p_upload.set_defaults(func=cmd_upload)

    args = parser.parse_args()
    args.func(get_client(), args)


if __name__ == "__main__":
    main()
