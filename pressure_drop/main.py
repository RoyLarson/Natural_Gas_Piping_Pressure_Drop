from .piping_pressure_drop import (SimpleNG, 
                                    Path, Pipe, GasFlow)

if __name__ == '__main__':
    fluid = SimpleNG(molecular_weight = 27)
    path = Path.from_list_of_lengths_and_angles([100], [-45])
    pipe = Pipe(12, .01625, path)
    flow = GasFlow(rate_mscfd= 100000, pipe= pipe, fluid=fluid, dp_with_flow=True)
