defmodule Bellend.Tokenizer do
  defstruct [:tokenizer]

  alias Tokenizers.Encoding
  alias Tokenizers.Tokenizer

  def new(model_id) do
    {:ok, tokenizer} = Tokenizer.from_pretrained(model_id)
    Tokenizer.add_special_tokens(tokenizer, ["<|endoftext|>"])

    %__MODULE__{tokenizer: tokenizer}
  end

  def encode(%{tokenizer: tokenizer}, text) do
    {:ok, encoding} = Tokenizer.encode(tokenizer, text, add_special_tokens: true)

    Encoding.get_ids(encoding)
  end

  def decode(%{tokenizer: tokenizer}, ids) do
    {:ok, tokens} = Tokenizer.decode(tokenizer, ids, skip_special_tokens: false)

    tokens
  end
end
