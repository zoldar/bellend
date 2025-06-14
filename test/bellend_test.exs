defmodule BellendTest do
  use ExUnit.Case
  doctest Bellend

  test "greets the world" do
    assert Bellend.hello() == :world
  end
end
