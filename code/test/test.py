import json


def main(service):
    with open("sample_data.json", "r") as fh:
        sample_data = json.loads(fh.read())
    input_data = json.dumps(sample_data)

    # # Creating input data
    # print("Creating input data")
    # data = {"data": [[1, 2, 3, 4], [10, 9, 8, 7]]}
    # input_data = json.dumps(data)

    # Calling webservice
    print("Calling webservice")
    output_data = service.run(input_data)
    predictions = output_data.get("predict")
    assert type(predictions) == list


if __name__ == "__main__":
    main()
