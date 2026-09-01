import requests

from olfactory.pubchem import NoveltyStatus, PubChemClient


class FakeResponse:
    def __init__(self, status_code, payload=None, json_error=None):
        self.status_code = status_code
        self.payload = payload
        self.json_error = json_error

    def json(self):
        if self.json_error:
            raise self.json_error
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def make_client(responses):
    session = FakeSession(responses)
    sleeps = []
    client = PubChemClient(session=session, sleeper=sleeps.append)
    return client, session, sleeps


def test_no_request_is_made_before_consent():
    client, session, _ = make_client([])
    result = client.verify("CCO", consent=False)
    assert result.status is NoveltyStatus.UNVERIFIED
    assert result.error_code == "CONSENT_REQUIRED"
    assert session.calls == []


def test_http_200_with_cids_is_found_and_uses_stereo_isotope_identity():
    client, session, _ = make_client(
        [FakeResponse(200, {"IdentifierList": {"CID": [702]}})]
    )
    result = client.verify("CCO", consent=True)
    assert result.status is NoveltyStatus.FOUND
    assert result.cids == (702,)
    _, kwargs = session.calls[0]
    assert kwargs["params"]["identity_type"] == "same_stereo_isotope"
    assert kwargs["timeout"] == (3.05, 10.0)


def test_http_404_is_not_found():
    client, _, _ = make_client([FakeResponse(404, {"Fault": {"Code": "PUGREST.NotFound"}})])
    assert client.verify("CCO", consent=True).status is NoveltyStatus.NOT_FOUND


def test_http_202_is_unverified():
    client, _, _ = make_client([FakeResponse(202, {})])
    result = client.verify("CCO", consent=True)
    assert result.status is NoveltyStatus.UNVERIFIED
    assert result.error_code == "HTTP_202"


def test_http_429_retries_with_exponential_backoff():
    client, session, sleeps = make_client([FakeResponse(429, {}), FakeResponse(404, {})])
    result = client.verify("CCO", consent=True)
    assert result.status is NoveltyStatus.NOT_FOUND
    assert len(session.calls) == 2
    assert sleeps == [0.5]


def test_http_503_exhausts_two_retries_then_fails_closed():
    client, session, sleeps = make_client(
        [FakeResponse(503, {}), FakeResponse(503, {}), FakeResponse(503, {})]
    )
    result = client.verify("CCO", consent=True)
    assert result.status is NoveltyStatus.UNVERIFIED
    assert result.error_code == "HTTP_503"
    assert len(session.calls) == 3
    assert sleeps == [0.5, 1.0]


def test_timeout_is_unverified_after_retries():
    client, session, _ = make_client(
        [requests.Timeout(), requests.Timeout(), requests.Timeout()]
    )
    result = client.verify("CCO", consent=True)
    assert result.status is NoveltyStatus.UNVERIFIED
    assert result.error_code == "TIMEOUT"
    assert len(session.calls) == 3


def test_malformed_200_response_is_unverified():
    client, _, _ = make_client([FakeResponse(200, {"unexpected": True})])
    result = client.verify("CCO", consent=True)
    assert result.status is NoveltyStatus.UNVERIFIED
    assert result.error_code == "MALFORMED_RESPONSE"


def test_result_is_cached_for_repeated_identity_check():
    client, session, _ = make_client([FakeResponse(404, {})])
    first = client.verify("CCO", consent=True)
    second = client.verify("CCO", consent=True)
    assert first == second
    assert len(session.calls) == 1


def test_client_never_starts_more_than_four_requests_in_one_second():
    class Clock:
        def __init__(self):
            self.now = 0.0
            self.sleeps = []

        def __call__(self):
            return self.now

        def sleep(self, seconds):
            self.sleeps.append(seconds)
            self.now += seconds

    clock = Clock()
    session = FakeSession([FakeResponse(404, {}) for _ in range(5)])
    client = PubChemClient(session=session, clock=clock, sleeper=clock.sleep)
    for index in range(5):
        client.verify(f"CCO{index}", consent=True)
    assert len(session.calls) == 5
    assert clock.sleeps == [1.0]
