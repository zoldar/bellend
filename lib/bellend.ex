defmodule Bellend do
  @moduledoc """
  Documentation for `Bellend`.
  """

  import Nx.Defn

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

  def test_self_attention0() do
    inputs =
      Nx.tensor([
        # Your (x^1)
        [0.43, 0.15, 0.89],
        # journey (x^2)
        [0.55, 0.87, 0.66],
        # starts (x^3)
        [0.57, 0.85, 0.64],
        # with (x^4)
        [0.22, 0.58, 0.33],
        # one (x^5)
        [0.77, 0.25, 0.10],
        # step (x^6)
        [0.05, 0.80, 0.55]
      ])

    query = inputs[1] |> IO.inspect(label: :QUERY)

    {size, _} = inputs.shape

    attn_scores_2 =
      0..(size - 1)
      |> Enum.map(&Nx.dot(inputs[&1], query))
      |> Nx.stack()
      |> IO.inspect(label: :ATTN_SCORES_2)

    attn_wieghts_2_tmp = Nx.divide(attn_scores_2, Nx.sum(attn_scores_2))

    IO.inspect(attn_wieghts_2_tmp, label: :ATTN_WEIGHTS_2_NAIVE)

    attn_weights_2 = softmax(attn_scores_2)

    IO.inspect(attn_weights_2, label: :ATTN_WEIGHTS_2)

    context_vec_2 =
      0..(size - 1)
      |> Enum.map(&Nx.multiply(attn_weights_2[&1], inputs[&1]))
      |> Nx.stack()
      |> Nx.sum(axes: [0])

    IO.inspect(context_vec_2, label: :CONTEXT_VEC_2)

    :ok
  end

  def test_self_attention1() do
    inputs =
      Nx.tensor([
        # Your (x^1)
        [0.43, 0.15, 0.89],
        # journey (x^2)
        [0.55, 0.87, 0.66],
        # starts (x^3)
        [0.57, 0.85, 0.64],
        # with (x^4)
        [0.22, 0.58, 0.33],
        # one (x^5)
        [0.77, 0.25, 0.10],
        # step (x^6)
        [0.05, 0.80, 0.55]
      ])

    attn_scores = Nx.dot(inputs, Nx.transpose(inputs))

    IO.inspect(attn_scores, label: :ATTENTION_SCORES)

    attn_weights = Nx.transpose(softmax(attn_scores, axes: [0]))

    IO.inspect(attn_weights, label: :ATTENTION_WEIGHTS)

    context_vecs = Nx.dot(attn_weights, inputs)

    IO.inspect(context_vecs, label: :CONTEXT_VECS)

    :ok
  end

  def test_self_attention_with_weights0() do
    inputs =
      Nx.tensor([
        # Your (x^1)
        [0.43, 0.15, 0.89],
        # journey (x^2)
        [0.55, 0.87, 0.66],
        # starts (x^3)
        [0.57, 0.85, 0.64],
        # with (x^4)
        [0.22, 0.58, 0.33],
        # one (x^5)
        [0.77, 0.25, 0.10],
        # step (x^6)
        [0.05, 0.80, 0.55]
      ])

    key = Nx.Random.key(123)
    x_2 = inputs[1]
    {_, d_in} = inputs.shape
    d_out = 2

    {w_query, key} = Nx.Random.uniform(key, shape: {d_in, d_out})
    {w_key, key} = Nx.Random.uniform(key, shape: {d_in, d_out})
    {w_value, _key} = Nx.Random.uniform(key, shape: {d_in, d_out})

    query_2 = Nx.dot(x_2, w_query)

    IO.inspect(query_2, label: :query_2)

    :ok
  end

  defn softmax(t, opts \\ []) do
    Nx.exp(t) / Nx.sum(Nx.exp(t), opts)
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
