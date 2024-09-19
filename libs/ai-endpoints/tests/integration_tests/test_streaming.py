import time

from langchain_nvidia_ai_endpoints import ChatNVIDIA


def test_ttft(chat_model: str, mode: dict) -> None:
    # we had an issue where streaming took a long time to start. the issue
    # was all streamed results were collected before yielding them to the
    # user. this test tries to detect the incorrect behavior.
    #
    # warning:
    #   - this can false positive if the model itself is slow to start
    #   - this can false nagative if there is a delay after the first chunk
    #
    # potential mitigation for false negative is to check mean & stdev and
    # filter outliers.
    #
    # credit to Pouyan Rezakhani for finding this issue
    llm = ChatNVIDIA(model=chat_model, **mode)
    chunk_times = [time.time()]
    for chunk in llm.stream("Count to 1000 by 2s, e.g. 2 4 6 8 ...", max_tokens=512):
        chunk_times.append(time.time())
    ttft = chunk_times[1] - chunk_times[0]
    total_time = chunk_times[-1] - chunk_times[0]
    assert ttft < (
        total_time / 2
    ), "potential streaming issue, TTFT should be less than half of the total time"
