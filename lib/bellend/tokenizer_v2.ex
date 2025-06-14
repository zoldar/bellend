defmodule Bellend.TokenizerV2 do
  defstruct [:str_to_int, :int_to_str]

  @unknown "<|unk|>"
  @endoftext "<|endoftext|>"

  def build_vocab(text) do
    text
    |> preprocess()
    |> MapSet.new()
    |> Enum.sort()
    |> Enum.concat([@endoftext, @unknown])
    |> Enum.with_index()
    |> Map.new()
  end

  def new(vocab) do
    %__MODULE__{
      str_to_int: vocab,
      int_to_str: Enum.into(vocab, %{}, fn {k, v} -> {v, k} end)
    }
  end

  def encode(t, text) do
    text
    |> preprocess()
    |> Enum.map(&(t.str_to_int[&1] || t.str_to_int[@unknown]))
  end

  def decode(t, ids) do
    ids
    |> Enum.map(&t.int_to_str[&1])
    |> Enum.join(" ")
    |> String.replace(~r/\s+([,.:;?!"()'])/, "\\1")
  end

  defp preprocess(text) do
    text
    |> String.split(~r/([,.:;?_!"()']|--|\s)/, include_captures: true)
    |> Enum.reject(&(String.trim(&1) == ""))
  end
end
