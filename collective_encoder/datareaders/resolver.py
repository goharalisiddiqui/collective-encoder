def get_datareader(datareader_Type: str):
    if datareader_Type == 'XTC':
        from collective_encoder.datareaders.xtc import XTCReader as DataReader
    elif datareader_Type == 'XTC_CHUNKS':
        from collective_encoder.datareaders.xtc_chunks import XTCChunksReader as DataReader
    elif datareader_Type == 'XTC_CHUNKS_CG':
        from collective_encoder.datareaders.xtc_chunks_cg import XTCChunksCGReader as DataReader
    else:
        raise ValueError("Unknown datareader type: " + datareader_Type)
    return DataReader