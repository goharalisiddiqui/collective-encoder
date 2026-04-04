def get_datareader(datareader_type: str):
    if datareader_type == 'XTC':
        from collective_encoder.datareaders.xtc import XTCReader as DataReader
    elif datareader_type == 'XTC_CHUNKS':
        from collective_encoder.datareaders.xtc_chunks import XTCChunksReader as DataReader
    elif datareader_type == 'XTC_CHUNKS_CG':
        from collective_encoder.datareaders.xtc_chunks_cg import XTCChunksCGReader as DataReader
    elif datareader_type == 'PLUMED_OUTPUT':
        from collective_encoder.datareaders.plumed_output import PlumedOutputReader as DataReader
    else:
        raise ValueError("Unknown datareader type: " + datareader_type)
    return DataReader