# core/core_storage.py
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from azure.identity import DefaultAzureCredential
import datetime

def get_blob_service_client_from_connstr(conn_str):
    return BlobServiceClient.from_connection_string(conn_str)

def get_blob_service_client_from_managed(endpoint_blob):
    cred = DefaultAzureCredential()
    return BlobServiceClient(account_url=endpoint_blob, credential=cred)

def upload_blob_from_bytes(cfg, container_name, blob_name, data_bytes, overwrite=True):
    if cfg.get('blob_connection_string'):
        bsc = get_blob_service_client_from_connstr(cfg['blob_connection_string'])
    elif cfg.get('blob_account_url'):
        bsc = get_blob_service_client_from_managed(cfg['blob_account_url'])
    else:
        raise RuntimeError("Proporciona 'blob_connection_string' o 'blob_account_url' en cfg")

    container_client = bsc.get_container_client(container_name)
    try:
        container_client.create_container()
    except Exception:
        pass

    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(data_bytes, overwrite=overwrite)
    return blob_client.url

def download_blob_to_bytes(cfg, container_name, blob_name):
    if cfg.get('blob_connection_string'):
        bsc = get_blob_service_client_from_connstr(cfg['blob_connection_string'])
    else:
        bsc = get_blob_service_client_from_managed(cfg['blob_account_url'])
    blob_client = bsc.get_blob_client(container=container_name, blob=blob_name)
    stream = blob_client.download_blob()
    return stream.readall()

def generate_sas_url(cfg, container_name, blob_name, expiry_minutes=60):
    if not cfg.get('account_name') or not cfg.get('account_key'):
        raise RuntimeError("Para generar SAS necesitas account_name y account_key en cfg")
    sas = generate_blob_sas(account_name=cfg['account_name'],
                            account_key=cfg['account_key'],
                            container_name=container_name,
                            blob_name=blob_name,
                            permission=BlobSasPermissions(read=True),
                            expiry=datetime.datetime.utcnow() + datetime.timedelta(minutes=expiry_minutes))
    account_url = cfg.get('blob_account_url') or f"https://{cfg['account_name']}.blob.core.windows.net"
    return f"{account_url}/{container_name}/{blob_name}?{sas}"
