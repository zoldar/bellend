defmodule Bellend do
  @moduledoc """
  Documentation for `Bellend`.
  """

  def test_tokenizer_v1() do
    alias Bellend.TokenizerV1

    text = File.read!("priv/the-verdict.txt")

    t =
      text
      |> TokenizerV1.build_vocab()
      |> TokenizerV1.new()

    phrase = """
    "It's the last he painted, you know," Mrs. Gisburn said with pardonable pride.
    """

    ids = TokenizerV1.encode(t, phrase)
    decoded = TokenizerV1.decode(t, ids)

    IO.inspect(ids, label: :ENCODED)
    IO.inspect(decoded, label: :DECODED)

    :ok
  end

  def test_tokenizer_v2() do
    alias Bellend.TokenizerV2

    text = File.read!("priv/the-verdict.txt")

    t =
      text
      |> TokenizerV2.build_vocab()
      |> TokenizerV2.new()

    phrase = """
    "It's the last he painted crap, you know," Mrs. Gisburn said with pardonable pride.
    <|endoftext|> Some other thought.
    """

    ids = TokenizerV2.encode(t, phrase)
    decoded = TokenizerV2.decode(t, ids)

    IO.inspect(ids, label: :ENCODED)
    IO.inspect(decoded, label: :DECODED)

    :ok
  end

  def test_tokenizer() do
    alias Bellend.Tokenizer

    t = Tokenizer.new("gpt2")

    phrase =
      "Hello, do you like tea? <|endoftext|> In the sunlit terraces " <>
        "of someunknownPlace."

    encoded = Tokenizer.encode(t, phrase)
    decoded = Tokenizer.decode(t, encoded)

    IO.inspect(encoded, label: :ENCODED)
    IO.inspect(decoded, label: :DECODED)
  end

  def test_data_sampling() do
    alias Bellend.GPTDatasetV1

    text = File.read!("priv/the-verdict.txt")

    d =
      GPTDatasetV1.new(text, "gpt2",
        batch_size: 8,
        max_length: 4,
        stride: 4,
        drop_last?: true
      )

    d |> Enum.take(1) |> IO.inspect()
  end

  def test_embedding0() do
    input_ids = Nx.tensor([2, 3, 5, 3])
    vocab_size = 6
    output_dim = 3

    embedding(input_ids, vocab_size, output_dim)
  end

  def test_embedding1() do
    alias Bellend.GPTDatasetV1

    text = File.read!("priv/the-verdict.txt")
    max_length = 4

    d =
      GPTDatasetV1.new(text, "gpt2",
        batch_size: 8,
        max_length: max_length,
        stride: max_length
      )

    [{inputs, _targets}] = Enum.take(d, 1)

    IO.inspect(inputs, label: :INPUTS)

    vocab_size = 50257
    output_dim = 256

    token_embeddings = embedding(inputs, vocab_size, output_dim)

    IO.inspect(token_embeddings, label: :TOKEN_EMBEDDINGS)

    pos_embeddings = embedding(Nx.iota({max_length}), vocab_size, output_dim)

    IO.inspect(pos_embeddings, label: :POS_EMBEDDINGS)

    # INPUT EMBEDDINGS
    Nx.add(token_embeddings, pos_embeddings)
  end

  defp embedding(input, vocab_size, output_dim) do
    {init_fn, predict_fn} =
      Axon.input("input")
      |> Axon.embedding(vocab_size, output_dim)
      |> Axon.build(seed: 123)

    params = init_fn.(input, Axon.ModelState.empty())
    predict_fn.(params, input)
  end
end
