defmodule Bellend.GPTDatasetV1 do
  def new(text, model_id, opts) do
    batch_size = Keyword.get(opts, :batch_size, 4)
    max_length = Keyword.get(opts, :max_length, 256)
    stride = Keyword.get(opts, :stride, 128)
    drop_last? = Keyword.get(opts, :drop_last?, true)

    t = Bellend.Tokenizer.new(model_id)

    input_ids = Bellend.Tokenizer.encode(t, text)

    batch_leftover =
      if drop_last? do
        :discard
      else
        :repeat
      end

    input_batches =
      to_batched(
        input_ids,
        max_length,
        batch_size,
        stride,
        batch_leftover
      )

    target_batches =
      input_ids
      |> Enum.drop(1)
      |> to_batched(
        max_length,
        batch_size,
        stride,
        batch_leftover
      )

    Stream.zip(input_batches, target_batches)
  end

  defp to_batched(ids, max_length, batch_size, stride, leftover) do
    ids
    |> Enum.chunk_every(max_length, stride, List.duplicate(0, max_length))
    |> Nx.tensor()
    |> Nx.to_batched(batch_size, leftover: leftover)
  end
end
